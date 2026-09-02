# SPAR evaluation entry point for ECCV 2026.
# Copyright (c) 2025 WildRayZer evaluation script.
# Adapted for SPAR; see THIRD_PARTY_NOTICES.md.
# D-RE10K-Mask evaluation for SPAR
#
#   - Sparse-view NVS: v=2, 3, 4 input views, 6 target views
#   - Motion mask accuracy: mIoU, Recall (vs human-verified annotations)
#
#   python test_spar.py \
#       -c configs/spar/spar.yaml \
#       --checkpoint ./checkpoints/spar.pt \
#       --test_root ./datasets/Dynamic-RE10K/test_zip \
#       --output_dir ./experiments/test_results/spar \
#
#   for v in 2 3 4; do
#       python test_spar.py \
#           -c configs/spar/spar.yaml \
#           --checkpoint ./checkpoints/spar.pt \
#           --num_input_views $v \
#           --output_dir ./experiments/test_results/spar_v${v}
#   done

import argparse
import datetime
import io
import json
import math
import os
import pickle
import sys
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from data.dataset_dre10k_test import DRE10KTestDataset
from easydict import EasyDict as edict
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image as PILImage
from skimage.metrics import structural_similarity
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


SEMANTIC_LABELS = [
    'wall', 'floor', 'cabinet', 'bed', 'chair',
    'apple', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator',
    'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurn'
]
NUM_SEMANTIC_CLASSES = len(SEMANTIC_LABELS) + 1  # +1 for background (class 0)

SEMANTIC_PALETTE = np.array([
    [  0,   0,   0],   # 0: background
    [174, 199, 232],   # 1: wall
    [152, 223, 138],   # 2: floor
    [ 31, 119, 180],   # 3: cabinet
    [140,  86,  75],   # 6: sofa
    [188, 189,  34],   # 5: chair
    [255, 187, 120],   # 4: bed
    [255, 152, 150],   # 7: table
    [214,  39,  40],   # 8: door
    [197, 176, 213],   # 9: window
    [148, 103, 189],   # 10: bookshelf
    [196, 156, 148],   # 11: picture
    [ 23, 190, 207],   # 12: counter
    [247, 182, 210],   # 13: desk
    [219, 219, 141],   # 14: curtain
    [255, 127,  14],   # 15: refrigerator
    [158, 218, 229],   # 16: shower curtain
    [ 44, 160,  44],   # 17: toilet
    [112, 128, 144],   # 18: sink
    [227, 119, 194],   # 19: bathtub
    [ 82,  84, 163],   # 20: otherfurn
], dtype=np.uint8)


def semantic_labels_to_color(labels_hw: np.ndarray) -> np.ndarray:
    """Map semantic class IDs to RGB colors."""
    H, W = labels_hw.shape
    color = SEMANTIC_PALETTE[labels_hw.clip(0, NUM_SEMANTIC_CLASSES - 1)]  # [H, W, 3]
    return color



def save_video(frames, video_path, fps=24):
    """Write RGB frames to an MP4 file with OpenCV."""
    import cv2
    
    if len(frames) == 0:
        raise ValueError("No frames to save")
        
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    
    try:
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
    finally:
        out.release()


def save_rendering_videos(
    video_rgb,           # [num_frames, 3, H, W]
    video_semantic,
    output_dir,
    scene_name,
    sample_idx,
    fps=24
):
    """Save RGB, semantic, and side-by-side rendering videos."""
    video_dir = os.path.join(output_dir, "videos", scene_name)
    os.makedirs(video_dir, exist_ok=True)
    
    num_frames = video_rgb.shape[0]
    
    rgb_frames = video_rgb.float().permute(0, 2, 3, 1).cpu().numpy()
    rgb_frames = (rgb_frames * 255).clip(0, 255).astype(np.uint8)
    
    rgb_video_path = os.path.join(video_dir, f"{sample_idx:04d}_rgb.mp4")
    save_video(rgb_frames, rgb_video_path, fps=fps)
    
    semantic_video_path = None
    combined_video_path = None
    
    if video_semantic is not None:
        semantic_frames = []
        for f_idx in range(num_frames):
            sem_labels = video_semantic[f_idx].cpu().numpy()
            sem_color = semantic_labels_to_color(sem_labels)
            semantic_frames.append(sem_color)
        semantic_frames = np.stack(semantic_frames, axis=0)
        
        semantic_video_path = os.path.join(video_dir, f"{sample_idx:04d}_semantic.mp4")
        save_video(semantic_frames, semantic_video_path, fps=fps)
        
        h, w = rgb_frames.shape[1:3]
        combined_frames = np.zeros((num_frames, h, w * 2, 3), dtype=np.uint8)
        combined_frames[:, :, :w, :] = rgb_frames
        combined_frames[:, :, w:, :] = semantic_frames
        
        combined_video_path = os.path.join(video_dir, f"{sample_idx:04d}_combined.mp4")
        save_video(combined_frames, combined_video_path, fps=fps)
    
    return rgb_video_path, semantic_video_path, combined_video_path


@torch.no_grad()
def decode_rendered_semantic(rendered_semantic, lseg_model, target_size, labelset=None):
    """Decode rendered LSeg features into open-vocabulary labels."""
    if labelset is None:
        labelset = SEMANTIC_LABELS
    
    rendered_semantic = rendered_semantic.float()
    
    with torch.cuda.amp.autocast(enabled=False):
        logits = lseg_model.decode_feature(rendered_semantic, labelset=labelset)
        logits_upsampled = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
        semantic_labels = torch.argmax(logits_upsampled, dim=1) + 1
    
    return semantic_labels


@torch.no_grad()
def render_video_with_semantic(
    actual_model,
    result,
    num_frames=60,
    loop_video=True,
    labelset=None
):
    """Render RGB and semantic videos with the available model interface."""
    if labelset is None:
        labelset = SEMANTIC_LABELS
        
    device = next(actual_model.parameters()).device
    
    if hasattr(actual_model, 'render_video_with_segmentation'):
        result = actual_model.render_video_with_segmentation(
            result,
            traj_type="interpolate",
            num_frames=num_frames,
            loop_video=loop_video,
            order_poses=False,
            labels=labelset
        )
        
        video_rgb = result.video_rendering[0]  # [num_frames, 3, H, W]
        video_semantic = result.video_segmentation[0] if hasattr(result, 'video_segmentation') else None
        
        return video_rgb, video_semantic
    
    elif hasattr(actual_model, 'render_video'):
        result = actual_model.render_video(
            result,
            traj_type="interpolate",
            num_frames=num_frames,
            loop_video=loop_video,
            order_poses=False
        )
        
        video_rgb = result.video_rendering[0]  # [num_frames, 3, H, W]
        
        video_semantic = None
        if hasattr(result, 'video_semantic') and result.video_semantic is not None:
            video_sem_features = result.video_semantic[0]  # [num_frames, 512, H//2, W//2]
            
            lseg_model = getattr(actual_model, 'lseg_model', None)
            if lseg_model is not None:
                semantic_list = []
                target_h = video_rgb.shape[-2]
                target_w = video_rgb.shape[-1]
                
                for f_idx in range(video_sem_features.shape[0]):
                    sem_feat = video_sem_features[f_idx:f_idx+1].to(device)
                    sem_labels = decode_rendered_semantic(
                        sem_feat,
                        lseg_model,
                        target_size=(target_h, target_w),
                        labelset=labelset
                    )
                    semantic_list.append(sem_labels[0].cpu())
                
                video_semantic = torch.stack(semantic_list, dim=0)  # [num_frames, H, W]
        
        return video_rgb, video_semantic
    
    elif hasattr(actual_model, 'render_images_video'):
        scene_tokens = getattr(result, 'scene_tokens', None)
        if scene_tokens is None:
            raise ValueError("No scene_tokens found in result for render_images_video")
        
        c2w = result.input.c2w if hasattr(result, 'input') and hasattr(result.input, 'c2w') else None
        fxfycxcy = result.input.fxfycxcy if hasattr(result, 'input') and hasattr(result.input, 'fxfycxcy') else None
        
        if c2w is None or fxfycxcy is None:
            raise ValueError("Missing camera parameters for video rendering")
        
        video_result = actual_model.render_images_video(
            scene_tokens, c2w, fxfycxcy, 
            normalized=True
        )
        
        video_rgb = video_result.rendered_images_video[0]
        
        video_semantic = None
        if hasattr(video_result, 'video_semantic') and video_result.video_semantic is not None:
            video_sem_features = video_result.video_semantic[0]
            
            lseg_model = getattr(actual_model, 'lseg_model', None)
            if lseg_model is not None:
                semantic_list = []
                target_h = video_rgb.shape[-2]
                target_w = video_rgb.shape[-1]
                
                for f_idx in range(video_sem_features.shape[0]):
                    sem_feat = video_sem_features[f_idx:f_idx+1].to(device)
                    sem_labels = decode_rendered_semantic(
                        sem_feat,
                        lseg_model,
                        target_size=(target_h, target_w),
                        labelset=labelset
                    )
                    semantic_list.append(sem_labels[0].cpu())
                
                video_semantic = torch.stack(semantic_list, dim=0)
        
        return video_rgb, video_semantic
    
    else:
        raise NotImplementedError("Model does not support video rendering")


@torch.no_grad()
def generate_semantic_predictions(
    result,
    actual_model,
    labelset=None,
    batch_idx=0,
):
    """Generate semantic predictions for input and rendered target views."""
    if labelset is None:
        labelset = SEMANTIC_LABELS

    b = batch_idx
    has_lseg = hasattr(actual_model, 'lseg_model') and actual_model.lseg_model is not None

    if not has_lseg:
        return None, None

    lseg_model = actual_model.lseg_model

    input_images = result.input.image[b]  # [V_input, 3, H, W]
    v_input, c, h, w = input_images.shape

    input_sem_features = lseg_model.extract_features(input_images)
    input_logits = lseg_model.decode_feature(input_sem_features, labelset=labelset)
    input_sem_labels = torch.argmax(input_logits, dim=1) + 1  # [V_in, H, W], 1~N

    target_sem_labels = None
    if hasattr(result, 'render_semantic') and result.render_semantic is not None:
        target_h = result.target.image.shape[-2]
        target_w = result.target.image.shape[-1]
        v_target = result.target.image.shape[1]

        rendered_sem = result.render_semantic  # [B*V_t, D_sem, H/2, W/2]
        start = b * v_target
        end = start + v_target
        rendered_sem_b = rendered_sem[start:end]  # [V_t, D_sem, H/2, W/2]

        target_logits = lseg_model.decode_feature(rendered_sem_b, labelset=labelset)
        target_sem_labels = torch.argmax(target_logits, dim=1) + 1  # [V_t, H, W]

    return input_sem_labels, target_sem_labels



@torch.no_grad()
def compute_masked_psnr(gt, pred, mask):
    """
    Masked PSNR (Eq. 3-4 in paper).

    Args:
        gt: [C, H, W], values [0, 1]
        pred: [C, H, W], values [0, 1]
        mask: [H, W], values [0, 1] (real-valued)
    Returns:
        psnr: scalar float
    """
    gt = gt.detach().cpu().float().clamp(0, 1)
    pred = pred.detach().cpu().float().clamp(0, 1)
    mask = mask.detach().cpu().float()

    if mask.sum() < 1.0:
        return float("nan")

    # [C, H, W] * [1, H, W] -> sum / (C * mask_sum)
    diff_sq = (gt - pred) ** 2  # [C, H, W]
    mse_per_pixel = diff_sq.mean(dim=0)
    mse_masked = (mse_per_pixel * mask).sum() / mask.sum()

    if mse_masked < 1e-10:
        return 100.0

    psnr = -10.0 * torch.log10(mse_masked)
    return psnr.item()


@torch.no_grad()
def compute_masked_ssim(gt, pred, mask):
    """
    Masked SSIM (Eq. 5-6 in paper).


    Args:
        gt: [C, H, W], values [0, 1]
        pred: [C, H, W], values [0, 1]
        mask: [H, W], values [0, 1]
    Returns:
        ssim: scalar float
    """
    gt = gt.detach().cpu().float().clamp(0, 1)
    pred = pred.detach().cpu().float().clamp(0, 1)
    mask = mask.detach().cpu().float()

    if mask.sum() < 1.0:
        return float("nan")

    gt_np = gt.numpy()
    pred_np = pred.numpy()

    # compute full SSIM map
    _, ssim_map = structural_similarity(
        gt_np,
        pred_np,
        win_size=11,
        gaussian_weights=True,
        sigma=1.5,
        channel_axis=0,
        data_range=1.0,
        full=True,
    )
    ssim_map = torch.from_numpy(np.asarray(ssim_map)).float().mean(dim=0)  # [H, W], CPU

    sh, sw = ssim_map.shape
    mh, mw = mask.shape
    if sh != mh or sw != mw:
        mask_for_ssim = F.adaptive_avg_pool2d(
            mask.unsqueeze(0).unsqueeze(0), (sh, sw)
        ).squeeze()
    else:
        mask_for_ssim = mask

    masked_ssim = (ssim_map * mask_for_ssim).sum() / (mask_for_ssim.sum() + 1e-8)
    return masked_ssim.item()


@torch.no_grad()
def compute_masked_lpips(gt, pred, mask, lpips_fn):
    """
    Masked LPIPS (Eq. 7 in paper).


    Args:
        gt: [1, C, H, W], values [0, 1]
        pred: [1, C, H, W], values [0, 1]
        mask: [H, W], values [0, 1]
        lpips_fn: LPIPS model with spatial=True
    Returns:
        lpips_val: scalar float
    """
    mask_cpu = mask.detach().cpu().float()
    if mask_cpu.sum() < 1.0:
        return float("nan")

    lpips_map = lpips_fn(
        gt * 2.0 - 1.0,
        pred * 2.0 - 1.0,
        normalize=False,
    )  # [1, 1, H', W']

    lpips_map = lpips_map.detach().cpu().squeeze().float()  # [H', W'], CPU
    lh, lw = lpips_map.shape

    mh, mw = mask_cpu.shape
    if lh != mh or lw != mw:
        mask_down = F.adaptive_avg_pool2d(
            mask_cpu.unsqueeze(0).unsqueeze(0), (lh, lw)
        ).squeeze()
    else:
        mask_down = mask_cpu

    masked_lpips = (lpips_map * mask_down).sum() / (mask_down.sum() + 1e-8)
    return masked_lpips.item()


@torch.no_grad()
def compute_full_image_metrics(gt, pred, lpips_fn_nospatial):
    """Compute full-image PSNR, SSIM, and LPIPS."""
    gt_f = gt.detach().float().clamp(0, 1)
    pred_f = pred.detach().float().clamp(0, 1)

    # PSNR
    mse = ((gt_f - pred_f) ** 2).mean()
    psnr = -10.0 * torch.log10(mse.clamp(min=1e-10)).item()

    ssim_val = structural_similarity(
        gt_f.cpu().numpy(),
        pred_f.cpu().numpy(),
        win_size=11,
        gaussian_weights=True,
        channel_axis=0,
        data_range=1.0,
    )

    lpips_val = lpips_fn_nospatial(
        gt_f.unsqueeze(0) * 2.0 - 1.0,
        pred_f.unsqueeze(0) * 2.0 - 1.0,
        normalize=False,
    ).item()

    return psnr, ssim_val, lpips_val


# 3. Motion Mask Metrics (mIoU, Recall)

@torch.no_grad()
def compute_mask_metrics(pred_mask, gt_mask, threshold=0.5):
    """Compute mIoU, recall, precision, and F1 for a dynamic-region mask."""
    pred_bin = (pred_mask > threshold).float()
    gt_bin = (gt_mask > threshold).float()

    # Foreground (dynamic) class
    fg_inter = (pred_bin * gt_bin).sum()  # TP
    fg_union = ((pred_bin + gt_bin) > 0).float().sum()
    fg_iou = (fg_inter + 1e-6) / (fg_union + 1e-6)

    # Background (static) class
    bg_pred = 1.0 - pred_bin
    bg_gt = 1.0 - gt_bin
    bg_inter = (bg_pred * bg_gt).sum()
    bg_union = ((bg_pred + bg_gt) > 0).float().sum()
    bg_iou = (bg_inter + 1e-6) / (bg_union + 1e-6)

    miou = (fg_iou + bg_iou) / 2.0

    # Recall: TP / (TP + FN) = TP / GT_positive
    gt_pos = gt_bin.sum()
    if gt_pos < 1.0:
        recall = float("nan")
    else:
        recall = (fg_inter / gt_pos).item()

    # Precision: TP / (TP + FP) = TP / Pred_positive
    pred_pos = pred_bin.sum()
    if pred_pos < 1.0:
        precision = float("nan")
    else:
        precision = (fg_inter / pred_pos).item()

    # F1 Score: 2 * Precision * Recall / (Precision + Recall)
    if math.isnan(recall) or math.isnan(precision) or (recall + precision) < 1e-6:
        f1 = float("nan")
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "miou": miou.item(),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


@torch.no_grad()
def compute_mask_iou_recall(pred_mask, gt_mask, threshold=0.5):
    """Return mIoU and recall for compatibility with earlier evaluators."""
    metrics = compute_mask_metrics(pred_mask, gt_mask, threshold)
    return metrics["miou"], metrics["recall"]



def load_config(config_path, overrides=None):
    config = OmegaConf.load(config_path)
    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, cli)

    config = OmegaConf.to_container(config, resolve=True)
    config = edict(config)

    if not hasattr(config, "inference"):
        config.inference = edict()
    config.inference.if_inference = True
    config.evaluation = True

    return config




@torch.no_grad()
def save_visualizations(
    result, gt_masks, scene_name, output_dir, batch_idx=0,
    input_sem_labels=None, target_sem_labels=None,
):
    """Save RGB, dynamic-mask, and semantic visualizations for one scene."""
    b = batch_idx
    sn = scene_name[b] if isinstance(scene_name, (list, tuple)) else scene_name
    scene_dir = os.path.join(output_dir, sn)
    os.makedirs(scene_dir, exist_ok=True)

    rendered = result.render.float()            # [B, V_target, 3, H, W]
    target_imgs = result.target.image.float()   # [B, V_target, 3, H, W]
    input_imgs = result.input.image.float()     # [B, V_input, 3, H, W]
    target_idx = result.target_idx              # [B, V_target]
    input_idx = result.input_idx                # [B, V_input]

    def _to_uint8(tensor):
        """[C, H, W] or [H, W, C] float [0,1] → uint8 numpy [H, W, C]."""
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)  # CHW → HWC
        return (tensor.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

    inp = input_imgs[b]  # [V_input, 3, H, W]
    inp_list = [_to_uint8(inp[v]) for v in range(inp.shape[0])]
    inp_concat = np.concatenate(inp_list, axis=1)  # [H, V*W, 3]
    PILImage.fromarray(inp_concat).save(os.path.join(scene_dir, "input.png"))

    gt_strip = []
    pred_strip = []
    v_target = rendered.shape[1]
    for v_idx in range(v_target):
        gt_np = _to_uint8(target_imgs[b, v_idx])
        pred_np = _to_uint8(rendered[b, v_idx])
        gt_strip.append(gt_np)
        pred_strip.append(pred_np)

        single_compare = np.concatenate([gt_np, pred_np], axis=0)
        PILImage.fromarray(single_compare).save(
            os.path.join(scene_dir, f"gt_vs_pred_view_{v_idx}.png")
        )

    gt_row = np.concatenate(gt_strip, axis=1)
    pred_row = np.concatenate(pred_strip, axis=1)
    comparison = np.concatenate([gt_row, pred_row], axis=0)
    PILImage.fromarray(comparison).save(os.path.join(scene_dir, "gt_vs_pred.png"))

    tgt_masks = gt_masks[b, target_idx[b]]  # [V_target, H, W]
    for v_idx in range(v_target):
        mask_np = (tgt_masks[v_idx].detach().cpu().numpy() * 255).astype(np.uint8)
        PILImage.fromarray(mask_np, mode="L").save(
            os.path.join(scene_dir, f"gt_mask_target_{v_idx}.png")
        )

    if hasattr(result, "motion_mask_target_soft") and result.motion_mask_target_soft is not None:
        pred_motion = result.motion_mask_target_soft[b]  # [V_target, 1, H, W]
        for v_idx in range(min(v_target, pred_motion.shape[0])):
            pred_m = pred_motion[v_idx, 0].detach().cpu().float()  # [H, W]
            gt_m = tgt_masks[v_idx].detach().cpu().float()

            pred_mask_np = (pred_m.clamp(0, 1).numpy() * 255).astype(np.uint8)
            PILImage.fromarray(pred_mask_np, mode="L").save(
                os.path.join(scene_dir, f"pred_mask_target_{v_idx}.png")
            )

            pred_bin = (pred_m > 0.5).float()
            gt_bin = (gt_m > 0.5).float()
            H, W = pred_bin.shape

            base_img = _to_uint8(target_imgs[b, v_idx]).astype(np.float32)
            overlay = base_img.copy()

            tp = (pred_bin * gt_bin).numpy().astype(bool)
            fn = ((1 - pred_bin) * gt_bin).numpy().astype(bool)
            fp = (pred_bin * (1 - gt_bin)).numpy().astype(bool)

            alpha = 0.5
            overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0, 255, 0], dtype=np.float32) * alpha
            overlay[fn] = overlay[fn] * (1 - alpha) + np.array([255, 0, 0], dtype=np.float32) * alpha
            overlay[fp] = overlay[fp] * (1 - alpha) + np.array([0, 0, 255], dtype=np.float32) * alpha

            PILImage.fromarray(overlay.astype(np.uint8)).save(
                os.path.join(scene_dir, f"mask_overlay_{v_idx}.png")
            )

    if hasattr(result, "motion_mask_input_soft") and result.motion_mask_input_soft is not None:
        pred_motion_inp = result.motion_mask_input_soft[b]  # [V_input, 1, H, W]
        inp_gt_masks = gt_masks[b, input_idx[b]]      # [V_input, H, W]
        for v_idx in range(pred_motion_inp.shape[0]):
            pred_m = pred_motion_inp[v_idx, 0].detach().cpu().float()
            pred_mask_np = (pred_m.clamp(0, 1).numpy() * 255).astype(np.uint8)
            PILImage.fromarray(pred_mask_np, mode="L").save(
                os.path.join(scene_dir, f"pred_mask_input_{v_idx}.png")
            )
            # GT mask for input view
            gt_mask_inp_np = (inp_gt_masks[v_idx].detach().cpu().numpy() * 255).astype(np.uint8)
            PILImage.fromarray(gt_mask_inp_np, mode="L").save(
                os.path.join(scene_dir, f"gt_mask_input_{v_idx}.png")
            )


    if input_sem_labels is not None:
        sem_inp_strips = []
        img_h, img_w = input_imgs.shape[-2], input_imgs.shape[-1]
        for v_idx in range(input_sem_labels.shape[0]):
            sem_np = input_sem_labels[v_idx].detach().cpu().numpy().astype(np.int32)
            sem_color = semantic_labels_to_color(sem_np)  # [H_sem, W_sem, 3] uint8

            if sem_color.shape[0] != img_h or sem_color.shape[1] != img_w:
                sem_color = np.array(PILImage.fromarray(sem_color).resize(
                    (img_w, img_h), resample=PILImage.NEAREST))

            PILImage.fromarray(sem_color).save(
                os.path.join(scene_dir, f"sem_input_{v_idx}.png")
            )
            sem_inp_strips.append(sem_color)

            base_img = _to_uint8(input_imgs[b, v_idx]).astype(np.float32)
            sem_overlay = base_img * 0.5 + sem_color.astype(np.float32) * 0.5
            PILImage.fromarray(sem_overlay.astype(np.uint8)).save(
                os.path.join(scene_dir, f"sem_input_overlay_{v_idx}.png")
            )

        if sem_inp_strips:
            sem_inp_all = np.concatenate(sem_inp_strips, axis=1)
            PILImage.fromarray(sem_inp_all).save(os.path.join(scene_dir, "sem_input_all.png"))

    if target_sem_labels is not None:
        sem_tgt_strips = []
        tgt_h, tgt_w = target_imgs.shape[-2], target_imgs.shape[-1]
        for v_idx in range(target_sem_labels.shape[0]):
            sem_np = target_sem_labels[v_idx].detach().cpu().numpy().astype(np.int32)
            sem_color = semantic_labels_to_color(sem_np)  # [H_sem, W_sem, 3] uint8

            if sem_color.shape[0] != tgt_h or sem_color.shape[1] != tgt_w:
                sem_color = np.array(PILImage.fromarray(sem_color).resize(
                    (tgt_w, tgt_h), resample=PILImage.NEAREST))

            PILImage.fromarray(sem_color).save(
                os.path.join(scene_dir, f"sem_target_{v_idx}.png")
            )
            sem_tgt_strips.append(sem_color)

            base_img = _to_uint8(rendered[b, v_idx]).astype(np.float32)
            sem_overlay = base_img * 0.5 + sem_color.astype(np.float32) * 0.5
            PILImage.fromarray(sem_overlay.astype(np.uint8)).save(
                os.path.join(scene_dir, f"sem_target_overlay_{v_idx}.png")
            )

        if sem_tgt_strips:
            sem_tgt_all = np.concatenate(sem_tgt_strips, axis=1)
            PILImage.fromarray(sem_tgt_all).save(os.path.join(scene_dir, "sem_target_all.png"))

        if len(sem_tgt_strips) == v_target:
            sem_row = np.concatenate(sem_tgt_strips, axis=1)
            three_row = np.concatenate([gt_row, pred_row, sem_row], axis=0)
            PILImage.fromarray(three_row).save(
                os.path.join(scene_dir, "gt_vs_pred_vs_sem.png")
            )

def setup_distributed():
    """Initialize torchrun DDP or select a single local device."""
    use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_ddp:
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        global_rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)

    return edict(
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        device=device,
        is_main=global_rank == 0,
        use_ddp=use_ddp,
    )


def gather_results_from_all_ranks(local_results, world_size, use_ddp):
    """Gather serialized per-scene results on rank zero."""
    if not use_ddp or world_size == 1:
        return local_results

    buffer = io.BytesIO()
    pickle.dump(local_results, buffer)
    data_bytes = buffer.getvalue()
    local_tensor = torch.ByteTensor(list(data_bytes)).to(dist.get_backend() != "nccl" and "cpu" or f"cuda:{dist.get_rank()}")

    local_size = torch.tensor([local_tensor.numel()], dtype=torch.long, device=local_tensor.device)
    size_list = [torch.zeros(1, dtype=torch.long, device=local_tensor.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)

    max_size = max(s.item() for s in size_list)
    padded = torch.zeros(max_size, dtype=torch.uint8, device=local_tensor.device)
    padded[: local_tensor.numel()] = local_tensor

    gather_list = [torch.zeros(max_size, dtype=torch.uint8, device=local_tensor.device) for _ in range(world_size)]
    dist.all_gather(gather_list, padded)

    if dist.get_rank() == 0:
        all_results = []
        for i, (g, s) in enumerate(zip(gather_list, size_list)):
            raw = bytes(g[: s.item()].cpu().tolist())
            all_results.extend(pickle.loads(raw))
        return all_results
    else:
        return []


# 6. Main

METRIC_DISPLAY_ORDER = [
    # Novel View Synthesis metrics (on target views)
    ("static_psnr", "Static PSNR ↑"),
    ("static_ssim", "Static SSIM ↑"),
    ("static_lpips", "Static LPIPS ↓"),
    ("full_psnr", "Full PSNR ↑"),
    ("full_ssim", "Full SSIM ↑"),
    ("full_lpips", "Full LPIPS ↓"),
    # Motion Mask metrics (on input views only)
    # CV-DRP raw output (before SAM2 refinement)
    ("mask_miou_input_raw", "Mask mIoU (Input, CV-DRP Raw) ↑"),
    ("mask_recall_input_raw", "Mask Recall (Input, CV-DRP Raw) ↑"),
    ("mask_precision_input_raw", "Mask Precision (Input, CV-DRP Raw) ↑"),
    ("mask_f1_input_raw", "Mask F1 (Input, CV-DRP Raw) ↑"),
    # SAM2 refined output
    ("mask_miou_input_refined", "Mask mIoU (Input, SAM2) ↑"),
    ("mask_recall_input_refined", "Mask Recall (Input, SAM2) ↑"),
    ("mask_precision_input_refined", "Mask Precision (Input, SAM2) ↑"),
    ("mask_f1_input_refined", "Mask F1 (Input, SAM2) ↑"),
]


def evaluate_batch(result, gt_masks, args, lpips_fn_spatial, lpips_fn_standard, scene_name):
    """Compute per-scene rendering and dynamic-mask metrics for one batch."""
    rendered = result.render.float()            # [B, V_target, 3, H, W]
    target_imgs = result.target.image.float()   # [B, V_target, 3, H, W]
    target_idx = result.target_idx              # [B, V_target]
    input_idx = result.input_idx                # [B, V_input]
    B = rendered.shape[0]

    batch_results = []

    for b in range(B):
        sn = scene_name[b] if isinstance(scene_name, (list, tuple)) else scene_name
        scene_metrics = {"scene_name": sn, "num_input_views": args.num_input_views}

        tgt_masks = gt_masks[b, target_idx[b]]  # [V_target, H, W]
        per_view_metrics = []

        static_ratios_per_view = []
        for v_idx in range(tgt_masks.shape[0]):
            s_mask = 1.0 - tgt_masks[v_idx].detach().cpu().float()
            static_ratios_per_view.append(s_mask.mean().item())
        scene_metrics["static_ratio"] = sum(static_ratios_per_view) / len(static_ratios_per_view) if static_ratios_per_view else 1.0

        for v_idx in range(rendered.shape[1]):
            gt_img = target_imgs[b, v_idx]
            pred_img = rendered[b, v_idx]
            motion_mask = tgt_masks[v_idx]
            static_mask = 1.0 - motion_mask

            view_metrics = {}

            if args.eval_static_region:
                view_metrics["static_psnr"] = compute_masked_psnr(gt_img, pred_img, static_mask)
                view_metrics["static_ssim"] = compute_masked_ssim(gt_img, pred_img, static_mask)
                view_metrics["static_lpips"] = compute_masked_lpips(
                    gt_img.unsqueeze(0), pred_img.unsqueeze(0), static_mask, lpips_fn_spatial,
                )

            if args.eval_full_image:
                f_psnr, f_ssim, f_lpips = compute_full_image_metrics(gt_img, pred_img, lpips_fn_standard)
                view_metrics["full_psnr"] = f_psnr
                view_metrics["full_ssim"] = f_ssim
                view_metrics["full_lpips"] = f_lpips

            per_view_metrics.append(view_metrics)

        if per_view_metrics and per_view_metrics[0]:
            for key in per_view_metrics[0].keys():
                vals = [m[key] for m in per_view_metrics if not math.isnan(m[key])]
                if vals:
                    scene_metrics[key] = sum(vals) / len(vals)

        if args.eval_motion_mask:
            inp_masks = gt_masks[b, input_idx[b]]  # [V_input, H, W]
            
            # --- CV-DRP Raw Output (before SAM2 refinement) ---
            if hasattr(result, "motion_mask_input_raw") and result.motion_mask_input_raw is not None:
                pred_raw = result.motion_mask_input_raw[b]  # [V_input, 1, H, W]
                raw_metrics = {"miou": [], "recall": [], "precision": [], "f1": []}
                
                for v_idx in range(pred_raw.shape[0]):
                    metrics = compute_mask_metrics(
                        pred_raw[v_idx, 0].float(), inp_masks[v_idx].float()
                    )
                    for k, v in metrics.items():
                        if not math.isnan(v):
                            raw_metrics[k].append(v)
                
                for k, vals in raw_metrics.items():
                    if vals:
                        scene_metrics[f"mask_{k}_input_raw"] = sum(vals) / len(vals)
            
            # --- SAM2 Refined Output ---
            if (
                args.sam2_initialized
                and hasattr(result, "motion_mask_input_refined")
                and result.motion_mask_input_refined is not None
            ):
                pred_refined = result.motion_mask_input_refined[b]  # [V_input, 1, H, W]
                refined_metrics = {"miou": [], "recall": [], "precision": [], "f1": []}
                
                for v_idx in range(pred_refined.shape[0]):
                    metrics = compute_mask_metrics(
                        pred_refined[v_idx, 0].float(), inp_masks[v_idx].float()
                    )
                    for k, v in metrics.items():
                        if not math.isnan(v):
                            refined_metrics[k].append(v)
                
                for k, vals in refined_metrics.items():
                    if vals:
                        scene_metrics[f"mask_{k}_input_refined"] = sum(vals) / len(vals)
            
            elif (
                args.sam2_initialized
                and hasattr(result, "motion_mask_input_soft")
                and result.motion_mask_input_soft is not None
            ):
                pred_soft = result.motion_mask_input_soft[b]
                refined_metrics = {"miou": [], "recall": [], "precision": [], "f1": []}
                
                for v_idx in range(pred_soft.shape[0]):
                    metrics = compute_mask_metrics(
                        pred_soft[v_idx, 0].float(), inp_masks[v_idx].float()
                    )
                    for k, v in metrics.items():
                        if not math.isnan(v):
                            refined_metrics[k].append(v)
                
                for k, vals in refined_metrics.items():
                    if vals:
                        scene_metrics[f"mask_{k}_input_refined"] = sum(vals) / len(vals)

        batch_results.append(scene_metrics)

    return batch_results


def save_and_print_summary(all_results, args, ckpt_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Per-scene JSON
    per_scene_path = os.path.join(output_dir, "per_scene_metrics.json")
    with open(per_scene_path, "w") as f:
        json.dump(all_results, f, indent=2)

    agg = defaultdict(list)
    static_ratios = []
    low_static_scenes = []
    for res in all_results:
        for key, _ in METRIC_DISPLAY_ORDER:
            if key in res and not math.isnan(res[key]):
                agg[key].append(res[key])
        
        if "static_ratio" in res:
            static_ratios.append(res["static_ratio"])
            if res["static_ratio"] < 0.05:
                low_static_scenes.append(res.get("scene_name", "unknown"))

    summary = {
        "num_scenes": len(all_results),
        "num_input_views": args.num_input_views,
        "num_target_views": args.num_target_views,
        "checkpoint": ckpt_path,
    }
    
    if static_ratios:
        summary["mean_static_ratio"] = sum(static_ratios) / len(static_ratios)
        summary["min_static_ratio"] = min(static_ratios)
        summary["max_static_ratio"] = max(static_ratios)
        summary["num_low_static_scenes"] = len(low_static_scenes)
    
    for key, vals in agg.items():
        summary[f"mean_{key}"] = sum(vals) / len(vals) if vals else float("nan")
        summary[f"std_{key}"] = float(np.std(vals)) if vals else float("nan")

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print(f"\n{'='*70}")
    print(f"  Evaluation Summary (v={args.num_input_views} input views)")
    print(f"  {len(all_results)} scenes evaluated")
    print(f"{'='*70}")
    
    # NVS metrics section
    print(f"\n  [Novel View Synthesis - Target Views]")
    print(f"  {'-'*50}")
    nvs_keys = ["static_psnr", "static_ssim", "static_lpips", 
                "full_psnr", "full_ssim", "full_lpips"]
    for key, display_name in METRIC_DISPLAY_ORDER:
        if key in nvs_keys and key in agg and agg[key]:
            mean_val = sum(agg[key]) / len(agg[key])
            std_val = float(np.std(agg[key]))
            print(f"  {display_name:35s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Motion Mask metrics section
    print(f"\n  [Motion Mask - Input Views Only]")
    print(f"  {'-'*50}")
    
    # CV-DRP Raw
    raw_keys = [k for k, _ in METRIC_DISPLAY_ORDER if "_raw" in k]
    if any(k in agg for k in raw_keys):
        print(f"  CV-DRP (Raw):")
        for key, display_name in METRIC_DISPLAY_ORDER:
            if "_raw" in key and key in agg and agg[key]:
                mean_val = sum(agg[key]) / len(agg[key])
                std_val = float(np.std(agg[key]))
                short_name = display_name.replace(" (Input, CV-DRP Raw)", "")
                print(f"    {short_name:30s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # SAM2 Refined
    refined_keys = [k for k, _ in METRIC_DISPLAY_ORDER if "_refined" in k]
    if any(k in agg for k in refined_keys):
        sam2_same_as_raw = True
        for raw_key in raw_keys:
            refined_key = raw_key.replace("_raw", "_refined")
            if raw_key in agg and refined_key in agg:
                raw_mean = sum(agg[raw_key]) / len(agg[raw_key]) if agg[raw_key] else 0
                ref_mean = sum(agg[refined_key]) / len(agg[refined_key]) if agg[refined_key] else 0
                if abs(raw_mean - ref_mean) > 1e-6:
                    sam2_same_as_raw = False
                    break
        
        if sam2_same_as_raw:
            print(f"  SAM2 Refined: (⚠ SAM2 not initialized, same as CV-DRP Raw)")
        else:
            print(f"  SAM2 Refined:")
        
        for key, display_name in METRIC_DISPLAY_ORDER:
            if "_refined" in key and key in agg and agg[key]:
                mean_val = sum(agg[key]) / len(agg[key])
                std_val = float(np.std(agg[key]))
                short_name = display_name.replace(" (Input, SAM2)", "")
                print(f"    {short_name:30s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\n{'='*70}")
    print(f"✓ Per-scene metrics: {per_scene_path}")
    print(f"✓ Summary: {summary_path}")

    # CSV
    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w") as f:
        header_keys = [k for k, _ in METRIC_DISPLAY_ORDER if k in agg]
        f.write("scene_name," + ",".join(header_keys) + "\n")
        for res in all_results:
            row = [res.get("scene_name", "")]
            for k in header_keys:
                val = res.get(k, float("nan"))
                row.append(f"{val:.6f}" if not math.isnan(val) else "nan")
            f.write(",".join(row) + "\n")
        avg_row = ["AVERAGE"]
        for k in header_keys:
            vals = agg[k]
            avg_row.append(f"{sum(vals)/len(vals):.6f}" if vals else "nan")
        f.write(",".join(avg_row) + "\n")
    print(f"✓ CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="SPAR D-RE10K-Mask Evaluation")
    parser.add_argument("--config", "-c", required=True, help="Model config YAML")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Checkpoint path. If empty, uses training.checkpoint_dir.")
    parser.add_argument(
        "--mask_source",
        type=str,
        default="raw",
        choices=["raw", "gt"],
        help=(
            "Dynamic mask used for rendering/evaluation. "
            "'gt' is an oracle diagnostic."
        ),
    )
    parser.add_argument("--test_root", type=str,
                        default="./datasets/Dynamic-RE10K/test_zip",
                        help="Test dataset root directory")
    parser.add_argument("--image_root", type=str, default="",
                        help="Optional decoded-frame root: <image_root>/<scene_id>/<frame>.png. "
                             "Also used to resolve '$Your Data Path$' in metadata image_path.")
    parser.add_argument("--prefer_metadata_image_path", action="store_true", default=True,
                        help="Prefer frame['image_path'] in metadata when available")
    parser.add_argument("--no_prefer_metadata_image_path", dest="prefer_metadata_image_path", action="store_false",
                        help="Disable metadata image_path and use heuristic decoded-frame paths only")
    parser.add_argument("--output_dir", type=str,
                        default="./experiments/test_results/spar_dre10k_mask",
                        help="Output directory for results")
    parser.add_argument("--num_input_views", type=int, default=2,
                        choices=[2, 3, 4],
                        help="Number of input views (2, 3, or 4)")
    parser.add_argument("--num_target_views", type=int, default=6,
                        help="Number of target views for evaluation")
    parser.add_argument("--fix_total_views", type=int, default=0,
                        help="Fix total views to match training (e.g., 8). "
                             "If set, num_target_views = fix_total_views - num_input_views. "
                             "This ensures temporal PE distribution matches training.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (1 recommended for precise metrics)")
    parser.add_argument("--single_gpu", action="store_true",
                        help="Force single-GPU mode (no DDP)")
    parser.add_argument("--eval_motion_mask", action="store_true", default=True,
                        help="Evaluate motion mask mIoU and Recall")
    parser.add_argument("--eval_static_region", action="store_true", default=True,
                        help="Report masked metrics for static regions (D-RE10K-Mask)")
    parser.add_argument("--eval_full_image", action="store_true", default=False,
                        help="Also report full-image metrics")
    parser.add_argument("--save_vis", action="store_true", default=True,
                        help="Save per-scene visualization images (input, GT vs pred, masks)")
    parser.add_argument("--no_save_vis", dest="save_vis", action="store_false",
                        help="Disable visualization saving for faster evaluation")
    parser.add_argument("--eval_semantic", action="store_true", default=True,
                        help="Enable semantic segmentation visualization (requires LSeg model)")
    parser.add_argument("--no_eval_semantic", dest="eval_semantic", action="store_false",
                        help="Disable semantic segmentation visualization")
    parser.add_argument("--view_idx_file", type=str, default="",
                        help="JSON file specifying per-scene context/target view indices. "
                             "Format: {scene_id: {context: [...], target: [...]}}. "
                             "When provided, only scenes in this file are evaluated, "
                             "and the fixed context/target indices override random selection.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (only used in single-GPU mode)")
    parser.add_argument("--render_video", action="store_true", default=False,
                        help="Render interpolation videos (RGB + semantic)")
    parser.add_argument("--num_frames", type=int, default=60,
                        help="Number of frames for video rendering")
    parser.add_argument("--fps", type=int, default=24,
                        help="FPS for video output")
    parser.add_argument("--loop_video", action="store_true", default=True,
                        help="Loop video back to start")
    parser.add_argument("--no_loop_video", dest="loop_video", action="store_false",
                        help="Disable video looping")
    args = parser.parse_args()

    if args.single_gpu:
        for env_key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(env_key, None)

    ddp = setup_distributed()
    device = ddp.device

    if args.fix_total_views > 0:
        actual_total_views = args.fix_total_views
        actual_target_views = args.fix_total_views - args.num_input_views
    else:
        actual_total_views = args.num_input_views + args.num_target_views
        actual_target_views = args.num_target_views
    
    config = load_config(args.config, overrides=[
        f"training.num_input_views={args.num_input_views}",
        f"training.num_target_views={actual_target_views}",
        f"training.num_views={actual_total_views}",
        "training.random_split=false",
        "training.batch_size_per_gpu=1",
        "training.grad_checkpoint_every=0",
        "inference.if_inference=true",
    ])
    config.inference = config.get("inference", edict())
    config.inference.if_inference = True
    config.evaluation = True

    ckpt_path = args.checkpoint or config.training.get("checkpoint_dir", "")

    if ddp.is_main:
        print(f"\n{'='*70}")
        print(f"  SPAR Evaluation on D-RE10K-Mask")
        print(f"  Config      : {args.config}")
        print(f"  Checkpoint  : {ckpt_path}")
        print(f"  Test root   : {args.test_root}")
        if args.image_root:
            print(f"  Image root  : {args.image_root}")
        print(f"  Prefer metadata image_path: {args.prefer_metadata_image_path}")
        print(f"  Input views : {args.num_input_views}")
        print(f"  Target views: {actual_target_views}")
        if args.fix_total_views > 0:
            print(f"  Total views : {actual_total_views} (fixed to match training)")
        print(f"  World size  : {ddp.world_size}")
        print(f"  Output      : {args.output_dir}")
        if args.render_video:
            print(f"  Video render: Enabled ({args.num_frames} frames @ {args.fps} fps, loop={args.loop_video})")
        else:
            print(f"  Video render: Disabled")
        print(f"{'='*70}\n")

    view_idx_dict = {}
    if args.view_idx_file and os.path.exists(args.view_idx_file):
        with open(args.view_idx_file, 'r') as f:
            view_idx_dict = json.load(f)
        view_idx_dict = {k: v for k, v in view_idx_dict.items() if v is not None}
        if ddp.is_main:
            print(f"✓ Loaded view_idx_file: {args.view_idx_file} ({len(view_idx_dict)} scenes)")
    elif args.view_idx_file:
        if ddp.is_main:
            print(f"⚠ view_idx_file not found: {args.view_idx_file}, using default view selection")

    test_dataset = DRE10KTestDataset(
        test_root=args.test_root,
        image_size=config.model.image_tokenizer.image_size,
        patch_size=config.model.image_tokenizer.patch_size,
        num_input_views=args.num_input_views,
        num_target_views=args.num_target_views,
        square_crop=config.training.get("square_crop", True),
        fix_total_views=args.fix_total_views,
        view_idx_dict=view_idx_dict,
        image_root=args.image_root,
        prefer_metadata_image_path=args.prefer_metadata_image_path,
    )
    
    if args.fix_total_views > 0:
        args.num_target_views = test_dataset.num_target_views

    if ddp.use_ddp:
        sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            sampler=sampler,
        )
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

    import importlib
    model_class_name = config.model.get("class_name", "model.spar.SPAR")
    module_name, cls_name = model_class_name.rsplit(".", 1)
    ModelClass = importlib.import_module(module_name).__dict__[cls_name]
    model = ModelClass(config).to(device)

    if ckpt_path:
        model.load_ckpt(ckpt_path)
        if ddp.is_main:
            print(f"✓ Loaded checkpoint from: {ckpt_path}")
    else:
        if ddp.is_main:
            print("⚠ Warning: No checkpoint specified, using random weights")

    if ddp.use_ddp:
        model = DDP(model, device_ids=[ddp.local_rank], find_unused_parameters=True)

    def _unwrap(m):
        return m.module if isinstance(m, DDP) else m

    _unwrap(model).eval()

    actual_model = _unwrap(model)
    try:
        _ = actual_model.dino_v3
        if ddp.is_main:
            print("✓ DINOv3 loaded → motion mask evaluation enabled")
    except Exception as e:
        if ddp.is_main:
            print(f"⚠ DINOv3 unavailable ({e}), motion mask eval may fail")

    # SAM2 refinement is intentionally disabled for this release.
    sam2_initialized = False
    if ddp.is_main:
        print("ℹ SAM2 refinement disabled → using raw CV-DRP masks")

    actual_model.mask_source_override = args.mask_source
    if ddp.is_main:
        print(f"✓ Evaluation mask source: {args.mask_source}")

    args.sam2_initialized = sam2_initialized

    has_lseg = hasattr(actual_model, 'lseg_model') and actual_model.lseg_model is not None
    use_semantic = has_lseg and args.eval_semantic
    labelset = None
    if use_semantic:
        lseg_config = config.model.get('lseg', {})
        labelset = lseg_config.get('labelset', SEMANTIC_LABELS)
        if isinstance(labelset, (list, tuple)):
            labelset = list(labelset)
        if ddp.is_main:
            print(f"✓ LSeg model found → semantic visualization enabled")
            print(f"  Labels: {labelset}")
    else:
        if ddp.is_main:
            if not has_lseg:
                print("ℹ Semantic visualization disabled (no LSeg model)")
            else:
                print("ℹ Semantic visualization disabled (--no_eval_semantic)")

    model.eval()

    if ddp.is_main:
        import lpips
        _ = lpips.LPIPS(net="vgg", spatial=True)
        _ = lpips.LPIPS(net="vgg")
    if ddp.use_ddp:
        dist.barrier()

    import lpips
    lpips_fn_spatial = lpips.LPIPS(net="vgg", spatial=True).to(device).eval()
    lpips_fn_standard = lpips.LPIPS(net="vgg").to(device).eval()

    if ddp.use_ddp:
        dist.barrier()

    os.makedirs(args.output_dir, exist_ok=True)
    local_results = []

    amp_dtype = getattr(torch, "bfloat16", torch.float16)

    if ddp.is_main:
        print(f"\nStarting evaluation on {len(test_dataset)} scenes "
              f"(~{len(test_loader)} batches per GPU)...\n")

    if ddp.use_ddp:
        sampler.set_epoch(0)

    pbar = tqdm(test_loader, desc=f"[Rank {ddp.global_rank}] Evaluating", disable=not ddp.is_main)

    for batch in pbar:
        scene_name = batch["scene_name"]
        if isinstance(scene_name, (list, tuple)):
            scene_name_str = scene_name[0]
        else:
            scene_name_str = scene_name

        data = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        gt_masks = data["binary_mask"]  # [B, V, H, W]

        try:
            with torch.no_grad(), torch.autocast(enabled=True, device_type="cuda", dtype=amp_dtype):
                result = _unwrap(model)(data, create_visual=False, render_video=False) \
                    if ddp.use_ddp else model(data, create_visual=False, render_video=False)

            batch_results = evaluate_batch(
                result, gt_masks, args, lpips_fn_spatial, lpips_fn_standard, scene_name,
            )
            local_results.extend(batch_results)

            if args.save_vis:
                B = result.render.shape[0]
                for b_idx in range(B):
                    inp_sem, tgt_sem = None, None
                    if use_semantic:
                        try:
                            inp_sem, tgt_sem = generate_semantic_predictions(
                                result, actual_model,
                                labelset=labelset, batch_idx=b_idx,
                            )
                        except Exception as sem_e:
                            print(f"[Rank {ddp.global_rank}] [WARN] Semantic prediction failed for {scene_name_str}: {sem_e}")

                    save_visualizations(
                        result, gt_masks, scene_name, args.output_dir,
                        batch_idx=b_idx,
                        input_sem_labels=inp_sem,
                        target_sem_labels=tgt_sem,
                    )
            
            if args.render_video:
                try:
                    B = result.render.shape[0]
                    for b_idx in range(B):
                        sn = scene_name[b_idx] if isinstance(scene_name, (list, tuple)) else scene_name
                        
                        video_rgb, video_semantic = render_video_with_semantic(
                            actual_model,
                            result,
                            num_frames=args.num_frames,
                            loop_video=args.loop_video,
                            labelset=labelset if use_semantic else SEMANTIC_LABELS
                        )
                        
                        if video_rgb is not None:
                            rgb_path, sem_path, combined_path = save_rendering_videos(
                                video_rgb,
                                video_semantic,
                                args.output_dir,
                                sn,
                                b_idx,
                                fps=args.fps
                            )
                            if ddp.is_main:
                                pbar.write(f"Saved video: {combined_path or rgb_path}")
                            
                except Exception as video_e:
                    import traceback
                    print(f"[Rank {ddp.global_rank}] [WARN] Video rendering failed for {scene_name_str}: {video_e}")
                    traceback.print_exc()

        except Exception as e:
            print(f"[Rank {ddp.global_rank}] [ERROR] Scene {scene_name_str}: {e}")
            continue

    if ddp.use_ddp:
        dist.barrier()

    all_results = gather_results_from_all_ranks(local_results, ddp.world_size, ddp.use_ddp)

    if ddp.is_main:
        seen = set()
        unique_results = []
        for r in all_results:
            sn = r.get("scene_name", "")
            if sn not in seen:
                seen.add(sn)
                unique_results.append(r)
        save_and_print_summary(unique_results, args, ckpt_path, args.output_dir)

    if ddp.use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
