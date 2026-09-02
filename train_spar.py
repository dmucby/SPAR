# SPAR training entry point for ECCV 2026.
# Copyright (c) 2025 WildRayZer implementation.
# Adapted for SPAR; see THIRD_PARTY_NOTICES.md.
# End-to-end joint training.
#
# Joint training mixes dynamic D-RE10K scenes with copy-paste augmented
# static RealEstate10K scenes. No offline pseudo-labels or alternating freezes
# are used.
# Usage:
#   torchrun --nproc_per_node=8 train_spar.py \
#     -c configs/spar/spar.yaml

import importlib
import os
import time
import contextlib
import copy
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup
from utils.metric_utils import visualize_intermediate_results
from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0


# Helpers

def _build_dataloader(dataset, batch_size, config, sampler=None):
    """Build a data loader with worker-safe prefetch settings."""
    num_workers = config.training.num_workers
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=False,
        drop_last=True,
        sampler=sampler,
    )
    # torch DataLoader requires prefetch_factor to be unset when num_workers == 0.
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = config.training.prefetch_factor
    return DataLoader(
        **dataloader_kwargs,
    )



def _sync_debug_timing_device(device):
    if not torch.cuda.is_available():
        return
    if isinstance(device, torch.device):
        if device.type != 'cuda':
            return
        torch.cuda.synchronize(device=device)
        return
    if isinstance(device, str) and device.startswith('cuda'):
        torch.cuda.synchronize(device=device)


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _next_batch(loader_iter, loader, sampler, epoch_counter, config,
                 dataset=None, batch_size=None, rebuild_loader=False,
                 on_new_epoch_fn=None):
    """Fetch a batch and safely advance the sampler and loader at epoch end."""
    try:
        data = next(loader_iter)
        return data, loader_iter, loader, epoch_counter, False
    except StopIteration:
        epoch_counter += 1
        if sampler is not None:
            sampler.set_epoch(epoch_counter)
        # Curriculum updates must run before rebuilding the loader.
        if on_new_epoch_fn is not None and dataset is not None:
            on_new_epoch_fn(dataset, epoch_counter)
        if rebuild_loader and dataset is not None and batch_size is not None:
            loader = _build_dataloader(dataset, batch_size, config, sampler)
        loader_iter = iter(loader)
        data = next(loader_iter)
        return data, loader_iter, loader, epoch_counter, True


def log_spar_metrics(loss_dict, cur_train_step, log_prefix="train"):
    return {f"{log_prefix}/{k}": v for k, v in loss_dict.items()}


def save_checkpoint(model, optimizer, lr_scheduler,
                    cur_train_step, cur_param_update_step, config):
    if isinstance(model, DDP):
        model_weights = model.module.state_dict()
    else:
        model_weights = model.state_dict()
    checkpoint = {
        "model": model_weights,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "fwdbwd_pass_step": cur_train_step,
        "param_update_step": cur_param_update_step,
    }
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        config.training.checkpoint_dir,
        f"ckpt_spar_{cur_train_step:012d}.pt",
    )
    torch.save(checkpoint, ckpt_path)
    print_rank0(f"Saved checkpoint at step {cur_train_step} → {os.path.abspath(ckpt_path)}")


# Init
config = init_config()
os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

ddp_info = init_distributed(seed=777, single_gpu=config.training.get("single_gpu", False))
if ddp_info.use_ddp:
    dist.barrier()

if ddp_info.is_main_process:
    init_wandb_and_backup(config)
if ddp_info.use_ddp:
    dist.barrier()

torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}
batch_size_per_gpu = config.training.batch_size_per_gpu

# Primary dataset: dynamic D-RE10K scenes
dre10k_dataset_name = config.training.get("dataset_name", "data.dataset_dre10k.DRE10KDataset")
dre10k_mod, dre10k_cls = dre10k_dataset_name.rsplit(".", 1)
DRE10KDataset = importlib.import_module(dre10k_mod).__dict__[dre10k_cls]
dre10k_dataset = DRE10KDataset(config)

if ddp_info.use_ddp:
    dre10k_sampler = DistributedSampler(dre10k_dataset)
else:
    dre10k_sampler = None
dre10k_loader = _build_dataloader(dre10k_dataset, batch_size_per_gpu, config, dre10k_sampler)
dre10k_loader_iter = iter(dre10k_loader)
dre10k_epoch = 0
print_rank0(f"[SPAR] D-RE10K dataset loaded: {len(dre10k_dataset)} scenes.")

# Secondary dataset: static RealEstate10K scenes for copy-paste augmentation
static_dataset = None
static_loader = None
static_loader_iter = None
static_sampler = None
static_epoch = 0

static_dataset_name = config.training.get("static_dataset_name", "")
static_dataset_path = config.training.get("static_dataset_path", "")
if static_dataset_name and static_dataset_path:
    try:
        static_config = copy.deepcopy(config)
        # The secondary dataset has a separate path but shares view settings.
        static_config.training.dataset_path = static_dataset_path
        s_mod, s_cls = static_dataset_name.rsplit(".", 1)
        StaticDataset = importlib.import_module(s_mod).__dict__[s_cls]
        static_dataset = StaticDataset(static_config)
        if ddp_info.use_ddp:
            static_sampler = DistributedSampler(static_dataset)
        static_loader = _build_dataloader(static_dataset, batch_size_per_gpu, config, static_sampler)
        static_loader_iter = iter(static_loader)
        print_rank0(f"[SPAR] Static RE10K dataset loaded: {len(static_dataset)} scenes.")
    except Exception as e:
        print_rank0(f"[SPAR] Failed to load static dataset: {e}")
        import traceback; traceback.print_exc()
        if config.training.get("copy_paste", {}).get("enabled", False):
            raise RuntimeError(
                "Copy-paste is enabled, so the static RealEstate dataset is required."
            ) from e
        static_dataset = None
else:
    print_rank0("[SPAR] No static_dataset configured — only D-RE10K will be used.")

# Copy-Paste Augmentor
copy_paste_augmentor = None
if config.training.get("copy_paste", {}).get("enabled", False):
    try:
        from data.copy_paste_augmentation import CopyPasteAugmentor
        copy_paste_augmentor = CopyPasteAugmentor(config)
        print_rank0("[SPAR] Copy-paste augmentor initialized.")
    except Exception as e:
        print_rank0(f"[SPAR] Copy-paste augmentor failed to init: {e}")
        import traceback; traceback.print_exc()
        raise RuntimeError(
            "Copy-paste is enabled but its COCO assets could not be initialized."
        ) from e

# Model
model_class_name = config.model.get("class_name", "model.spar.SPAR")
mod, cls = model_class_name.rsplit(".", 1)
ModelClass = importlib.import_module(mod).__dict__[cls]
model = ModelClass(config).to(ddp_info.device)

if copy_paste_augmentor is not None:
    model.copy_paste_augmentor = copy_paste_augmentor

# Load the SPAR reconstruction and CV-DRP initialization
pretrained_ckpt = config.training.get("pretrained_ckpt", "")
if pretrained_ckpt and os.path.exists(pretrained_ckpt):
    print_rank0(f"[SPAR] Loading pretrained weights from {pretrained_ckpt}")
    ckpt = torch.load(pretrained_ckpt, map_location="cpu", weights_only=True)
    model_state = ckpt.get("model", ckpt)
    status = model.load_compatible_state_dict(model_state, strict=False)
    print_rank0(f"  Missing keys ({len(status.missing_keys)}): {status.missing_keys[:15]}")
    print_rank0(f"  Unexpected keys ({len(status.unexpected_keys)}): {status.unexpected_keys[:15]}")
    del ckpt
    torch.cuda.empty_cache()
elif pretrained_ckpt:
    message = f"[SPAR] pretrained_ckpt not found at {pretrained_ckpt}"
    if not config.training.get("allow_train_from_scratch", False):
        raise FileNotFoundError(
            message + ". Set training.allow_train_from_scratch=true only for ablations."
        )
    print_rank0(message + " — explicitly training from scratch.")
else:
    if not config.training.get("allow_train_from_scratch", False):
        raise ValueError(
            "Paper reproduction requires training.pretrained_ckpt. "
            "Set training.allow_train_from_scratch=true only for ablations."
        )
    print_rank0("[SPAR] No pretrained_ckpt specified — explicitly training from scratch.")

# DINOv3 / CV-DRP / optional SAM2 initialization
use_cached_mask = config.training.get("mask_cache_dir", "") != ""

if not use_cached_mask:
    try:
        from model.dino_v3 import load_dinov3_from_local, DINOv3FeatureExtractor
        dino_config = config.model.get("dino_v3", {})
        dino_ckpt = dino_config.get(
            "checkpoint_path",
            "./datasets/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        )
        if os.path.exists(dino_ckpt):
            dino_arch = dino_config.get("arch", "vitb16")
            raw_dino = load_dinov3_from_local(
                dino_ckpt,
                arch=dino_arch,
                img_size=config.model.image_tokenizer.image_size,
                device=str(ddp_info.device),
            )
            dino_v3_model = DINOv3FeatureExtractor(raw_dino).to(ddp_info.device)
            model._dino_v3 = dino_v3_model
            print_rank0("[SPAR] DINOv3 loaded and shared with model (frozen).")
        else:
            print_rank0(f"[SPAR] WARNING: DINOv3 checkpoint not found at {dino_ckpt}")
    except Exception as e:
        print_rank0(f"[SPAR] DINOv3 loading failed: {e}")
        import traceback; traceback.print_exc()

    if hasattr(model, 'freeze_dino'):
        model.freeze_dino()

    freeze_cv_drp = bool(config.training.get('freeze_cv_drp', False))
    if freeze_cv_drp:
        model.freeze_cv_drp()
    else:
        model.unfreeze_cv_drp()

    # The paper uses SAM2 only for optional test-time mask refinement.
    # This opt-in switch is kept only for controlled ablations.
    enable_sam2_during_training = bool(
        config.training.get('enable_sam2_during_training', False)
    )
    if enable_sam2_during_training:
        sam2_config = config.model.get('sam2', {})
        if sam2_config and hasattr(model, 'init_sam2'):
            sam2_cfg_path = sam2_config.get('config', '')
            sam2_ckpt_path = sam2_config.get('checkpoint', '')
            if sam2_cfg_path and sam2_ckpt_path:
                model.init_sam2(
                    sam2_cfg_path,
                    sam2_ckpt_path,
                    device=str(ddp_info.device),
                )
else:
    print_rank0(f"[SPAR] Cached-mask mode: mask_cache_dir={config.training.mask_cache_dir}")
    print_rank0("[SPAR] Skipping DINOv3/CV-DRP/SAM2 initialization.")

# Optional cached-mask loader
import numpy as np

_mask_cache = {}  # scene_id -> numpy array [N_frames, H, W]

def load_cached_masks(data_batch, mask_cache_dir, image_size):
    """Load precomputed masks and attach them to a data batch."""
    if not mask_cache_dir or not os.path.isdir(mask_cache_dir):
        return

    scene_ids = data_batch.get('scene_name', None)
    frame_indices = data_batch.get('frame_indices', None)

    if scene_ids is None or frame_indices is None:
        return

    B, V = frame_indices.shape[:2]
    H = W = image_size
    masks = torch.zeros(B, V, 1, H, W, dtype=torch.float32)

    for b_idx in range(B):
        sid = scene_ids[b_idx]
        if sid not in _mask_cache:
            npz_path = os.path.join(mask_cache_dir, f"{sid}.npz")
            if os.path.exists(npz_path):
                loaded = np.load(npz_path)
                _mask_cache[sid] = loaded['masks']  # [N_frames, H_cache, W_cache]
            else:
                _mask_cache[sid] = None

        cached = _mask_cache[sid]
        if cached is None:
            continue

        for v_idx in range(V):
            fi = int(frame_indices[b_idx, v_idx])
            if fi < len(cached):
                mask_np = cached[fi].astype(np.float32) / 255.0  # [H_cache, W_cache]
                mask_t = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H_c, W_c]
                if mask_t.shape[1] != H or mask_t.shape[2] != W:
                    mask_t = torch.nn.functional.interpolate(
                        mask_t.unsqueeze(0), size=(H, W), mode='nearest'
                    ).squeeze(0)
                masks[b_idx, v_idx] = mask_t

    data_batch['cached_masks'] = masks

# DDP
def _get_model(m):
    return m.module if isinstance(m, DDP) else m

if ddp_info.use_ddp:
    model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=True)

# Optimizer & Scheduler
optimizer, optimized_param_dict, all_param_dict = create_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
)
optim_param_list = list(optimized_param_dict.values())

total_train_steps = config.training.train_steps
grad_accum_steps = config.training.grad_accum_steps
total_param_update_steps = total_train_steps
total_train_steps_real = total_train_steps * grad_accum_steps
total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps

scheduler_type = config.training.get("scheduler_type", "cosine")
lr_scheduler = create_lr_scheduler(
    optimizer,
    total_param_update_steps,
    config.training.warmup,
    scheduler_type=scheduler_type,
)

# Auto-resume
if config.training.get("resume_ckpt", "") != "":
    ckpt_load_path = config.training.resume_ckpt
else:
    ckpt_load_path = config.training.checkpoint_dir
reset_training_state = config.training.get("reset_training_state", False)
optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
    ckpt_load_path, model, optimizer, lr_scheduler, reset_training_state,
)

# Grad scaler
enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
scaler = torch.amp.GradScaler("cuda", enabled=enable_grad_scaler)
print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
if ddp_info.use_ddp:
    dist.barrier()

# Training Loop
copy_paste_prob = config.training.get("copy_paste", {}).get("prob", 0.5)
start_train_step = cur_train_step
model.train()
actual_model = _unwrap_model(model)
debug_timing_config = config.training.get("debug_timing", {})
debug_timing_enabled = bool(debug_timing_config.get("enabled", False))
debug_timing_print_every = int(debug_timing_config.get("print_every", config.training.print_every))

has_static = (static_loader is not None and copy_paste_augmentor is not None)
spatialvid_view_selector_config = config.training.get(
    "spatialvid_view_selector",
    config.training.get("view_selector", {}),
)
use_curriculum = spatialvid_view_selector_config.get("use_curriculum", False)

print_rank0(f"\n{'='*60}")
print_rank0(f"  SPAR Joint Training")
print_rank0(f"  Total train steps: {total_train_steps_real}")
print_rank0(f"  Param update steps: {total_param_update_steps}")
print_rank0(f"  Batch size per GPU: {batch_size_per_gpu}")
print_rank0(f"  Grad accumulation: {grad_accum_steps}")
print_rank0(f"  Total effective batch: {total_batch_size}")
print_rank0(f"  Learning rate: {config.training.lr}")
print_rank0(f"  Warmup steps: {config.training.warmup}")
print_rank0(f"  D-RE10K scenes: {len(dre10k_dataset)}")
print_rank0(f"  Static RE10K: {'YES (' + str(len(static_dataset)) + ' scenes)' if static_dataset else 'NO'}")
print_rank0(f"  Copy-paste prob: {copy_paste_prob}")
print_rank0(f"  Copy-paste augmentor: {'YES' if copy_paste_augmentor else 'NO'}")
print_rank0(f"{'='*60}\n")


while cur_train_step < total_train_steps_real:
    tic = time.perf_counter()
    iter_start = tic
    timing_stats = {
        "data_time": 0.0,
        "forward_time": 0.0,
        "sam3_feature_time": 0.0,
        "sam3_feature_calls": 0,
        "backward_time": 0.0,
        "optim_time": 0.0,
        "iter_time": 0.0,
    }

    # Select dynamic data or static data with copy-paste augmentation.
    use_static_copypaste = (
        has_static
        and torch.rand(1).item() < copy_paste_prob
    )

    if use_static_copypaste:
        # Static data does not use curriculum-based loader rebuilding.
        data, static_loader_iter, static_loader, static_epoch, epoch_changed = \
            _next_batch(
                static_loader_iter, static_loader, static_sampler, static_epoch,
                config, dataset=static_dataset, batch_size=batch_size_per_gpu,
                rebuild_loader=False,
            )
        if epoch_changed:
            print_rank0(f"[SPAR] Static RE10K completed epoch {static_epoch - 1}, "
                        f"starting epoch {static_epoch}")
        data_source = "static+cp"
    else:
        # Apply curriculum updates before rebuilding the D-RE10K loader.
        def _dre10k_on_new_epoch(ds, ep):
            if use_curriculum and hasattr(ds, 'update_iteration'):
                ds.update_iteration(cur_train_step)

        data, dre10k_loader_iter, dre10k_loader, dre10k_epoch, epoch_changed = \
            _next_batch(
                dre10k_loader_iter, dre10k_loader, dre10k_sampler, dre10k_epoch,
                config, dataset=dre10k_dataset, batch_size=batch_size_per_gpu,
                rebuild_loader=use_curriculum,
                on_new_epoch_fn=_dre10k_on_new_epoch,
            )
        if epoch_changed:
            print_rank0(f"[SPAR] D-RE10K completed epoch {dre10k_epoch - 1}, "
                        f"starting epoch {dre10k_epoch}")
        data_source = "dre10k"

    batch = {
        k: v.to(ddp_info.device) if isinstance(v, torch.Tensor) else v
        for k, v in data.items()
    }

    # Cached masks are available only for D-RE10K scenes.
    if use_cached_mask and data_source == "dre10k":
        mask_cache_dir = config.training.get("mask_cache_dir", "")
        image_size = config.model.image_tokenizer.image_size
        load_cached_masks(batch, mask_cache_dir, image_size)
    if 'cached_masks' in batch:
        batch['cached_masks'] = batch['cached_masks'].to(ddp_info.device)

    if debug_timing_enabled:
        _sync_debug_timing_device(ddp_info.device)
    timing_stats["data_time"] = time.perf_counter() - iter_start

    create_visual = (
        (cur_train_step - 1) == start_train_step
        or (cur_train_step % config.training.vis_every == 0)
    )
    render_video = create_visual and config.training.get("render_video", False)

    # Forward
    if debug_timing_enabled and hasattr(actual_model, 'reset_debug_timing'):
        actual_model.reset_debug_timing()
        _sync_debug_timing_device(ddp_info.device)
    forward_start = time.perf_counter()
    with torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        ret_dict = model(
            batch,
            create_visual=create_visual,
            render_video=render_video,
            iter=cur_train_step,
            # Copy-paste is enabled only for static RealEstate10K batches.
            is_copypaste_augmented=use_static_copypaste,
            paste_masks=None,
        )
    if debug_timing_enabled:
        _sync_debug_timing_device(ddp_info.device)
        timing_stats["forward_time"] = time.perf_counter() - forward_start
        if hasattr(actual_model, 'get_debug_timing'):
            model_timing = actual_model.get_debug_timing()
            timing_stats["sam3_feature_time"] = float(model_timing.get("sam3_feature_time", 0.0))
            timing_stats["sam3_feature_calls"] = int(model_timing.get("sam3_feature_calls", 0))

    # Backward
    update_grads = (
        (cur_train_step + 1) % grad_accum_steps == 0
        or cur_train_step == total_train_steps_real
    )
    no_sync_ctx = model.no_sync if ddp_info.use_ddp else lambda: contextlib.nullcontext()

    backward_start = time.perf_counter() if debug_timing_enabled else None
    if update_grads:
        scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    else:
        with no_sync_ctx():
            scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    if debug_timing_enabled:
        _sync_debug_timing_device(ddp_info.device)
        timing_stats["backward_time"] = time.perf_counter() - backward_start
    cur_train_step += 1

    total_grad_norm = 0.0

    if update_grads:
        optim_start = time.perf_counter() if debug_timing_enabled else None
        skip_optimizer_step = False

        if torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(ret_dict.loss_metrics.loss):
            print(f"NaN or Inf loss detected at step {cur_train_step}, skipping.")
            skip_optimizer_step = True
            ret_dict.loss_metrics.loss.data = torch.zeros_like(ret_dict.loss_metrics.loss)

        if not skip_optimizer_step:
            scaler.unscale_(optimizer)
            with torch.no_grad():
                for n, p in optimized_param_dict.items():
                    if p.requires_grad and p.grad is not None:
                        p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)

            if config.training.grad_clip_norm > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    optim_param_list, max_norm=config.training.grad_clip_norm,
                ).item()
                allowed = config.training.grad_clip_norm * config.training.get(
                    "allowed_gradnorm_factor", 5,
                )
                if total_grad_norm > allowed and cur_train_step > config.training.get(
                    "no_pass_steps", -1,
                ):
                    skip_optimizer_step = True
                    print(
                        f"WARNING: step {cur_train_step} grad norm "
                        f"{total_grad_norm:.2f} > {allowed:.2f}, skipping"
                    )

            if not skip_optimizer_step:
                scaler.step(optimizer)
                cur_param_update_step += 1

        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if debug_timing_enabled:
            _sync_debug_timing_device(ddp_info.device)
            timing_stats["optim_time"] = time.perf_counter() - optim_start

    if debug_timing_enabled:
        timing_stats["iter_time"] = time.perf_counter() - iter_start

    # Logging & Checkpointing
    if ddp_info.is_main_process:
        loss_dict = {
            k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()
        }

        # Console
        if (cur_train_step % config.training.print_every == 0) or (
            cur_train_step < 100 + start_train_step
        ):
            iter_time_for_print = timing_stats["iter_time"] if debug_timing_enabled else (time.perf_counter() - tic)
            s = f"[SPAR] [dre10k_ep={dre10k_epoch} static_ep={static_epoch}]"
            s += f" | Step {cur_train_step:>6d} (Update {cur_param_update_step:>6d})"
            s += f" | {iter_time_for_print:.2f}s | LR {optimizer.param_groups[0]['lr']:.6f}"
            s += f" | src={data_source}"
            if debug_timing_enabled and (cur_train_step % debug_timing_print_every == 0):
                s += (
                    f" | data {timing_stats['data_time']:.2f}s"
                    f" | fwd {timing_stats['forward_time']:.2f}s"
                    f" | sam3 {timing_stats['sam3_feature_time']:.2f}s"
                    f"/{timing_stats['sam3_feature_calls']}"
                    f" | bwd {timing_stats['backward_time']:.2f}s"
                    f" | opt {timing_stats['optim_time']:.2f}s"
                )
            s += "\n"
            for k, v in loss_dict.items():
                s += f"{k}: {v} | "
            print(s)

        # Wandb
        if (cur_train_step % config.training.wandb_log_every == 0) or (
            cur_train_step < 200 + start_train_step
        ):
            log_dict = {
                "iter": cur_train_step,
                "param_update_step": cur_param_update_step,
                "lr": optimizer.param_groups[0]["lr"],
                "iter_time": timing_stats["iter_time"] if debug_timing_enabled else (time.perf_counter() - tic),
                "grad_norm": total_grad_norm,
                "dre10k_epoch": dre10k_epoch,
                "static_epoch": static_epoch,
                "data_source": 1 if use_static_copypaste else 0,
            }
            if debug_timing_enabled:
                log_dict.update({
                    "data_time": timing_stats["data_time"],
                    "forward_time": timing_stats["forward_time"],
                    "sam3_feature_time": timing_stats["sam3_feature_time"],
                    "sam3_feature_calls": timing_stats["sam3_feature_calls"],
                    "backward_time": timing_stats["backward_time"],
                    "optim_time": timing_stats["optim_time"],
                })
            log_dict.update(log_spar_metrics(loss_dict, cur_train_step))
            wandb.log(log_dict, step=cur_train_step)

        # Checkpoint
        if (cur_train_step % config.training.checkpoint_every == 0) or (
            cur_train_step == total_train_steps_real
        ):
            save_checkpoint(
                model, optimizer, lr_scheduler,
                cur_train_step, cur_param_update_step, config,
            )

        # Visualization
        if create_visual:
            vis_path = os.path.join(
                config.training.checkpoint_dir, f"iter_{cur_train_step:08d}"
            )
            os.makedirs(vis_path, exist_ok=True)
            visualize_intermediate_results(vis_path, ret_dict)

            # Dynamic-region mask visualization
            if hasattr(ret_dict, "motion_mask_input_soft"):
                import torchvision
                for tag, mask_tensor in [
                    ("input", ret_dict.motion_mask_input_soft),
                    ("target", ret_dict.motion_mask_target_soft),
                ]:
                    mask_vis = mask_tensor[:1]  # [1, V, 1, H, W]
                    V = mask_vis.shape[1]
                    for v_idx in range(min(V, 4)):
                        torchvision.utils.save_image(
                            mask_vis[0, v_idx, 0].cpu().unsqueeze(0),
                            os.path.join(vis_path, f"motion_mask_{tag}_v{v_idx}.png"),
                        )

            torch.cuda.empty_cache()
            model.train()

    if create_visual:
        torch.cuda.empty_cache()
        if ddp_info.use_ddp:
            dist.barrier()


if ddp_info.use_ddp:
    dist.barrier()
    dist.destroy_process_group()
print_rank0("\n[SPAR] Training complete.")
