# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import torch
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
import torch.distributed as dist
import os
from rich import print
import traceback
from torch.nn.parallel import DistributedDataParallel as DDP


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)

def create_optimizer(model, weight_decay, learning_rate, betas):
    all_param_dict = {name: param for name, param in model.named_parameters()}

    if hasattr(model, 'get_parameter_groups'):
        raw_groups = model.get_parameter_groups(getattr(model, 'current_stage', None))
        optimizer_groups = []
        optimized_param_dict = {}
        added_ids = set()

        for group in raw_groups:
            params = []
            for param in group['params']:
                if (not param.requires_grad) or (id(param) in added_ids):
                    continue
                params.append(param)
                added_ids.add(id(param))

            if not params:
                continue

            decay_params, nodecay_params = [], []
            for param in params:
                if param.dim() == 1 or getattr(param, '_no_weight_decay', False):
                    nodecay_params.append(param)
                else:
                    decay_params.append(param)

            if decay_params:
                optimizer_groups.append({
                    'name': group['name'],
                    'params': decay_params,
                    'lr': group['lr'],
                    'weight_decay': weight_decay,
                })
            if nodecay_params:
                optimizer_groups.append({
                    'name': f"{group['name']}_nodecay",
                    'params': nodecay_params,
                    'lr': group['lr'],
                    'weight_decay': 0.0,
                })

        for name, param in model.named_parameters():
            if param.requires_grad and id(param) in added_ids:
                optimized_param_dict[name] = param

        optimizer = torch.optim.AdamW(optimizer_groups, betas=betas, foreach=True)
    else:
        optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}
        decay_params, nodecay_params = [], []
        for name, param in optimized_param_dict.items():
            if param.dim() == 1 or getattr(param, '_no_weight_decay', False):
                nodecay_params.append(param)
            else:
                decay_params.append(param)
        optimizer_groups = [
            {'name': 'default', 'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'name': 'default_nodecay', 'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
        ]
        optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate, betas=betas, foreach=True)
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        def get_module_name(name):
            parts = name.split('.')
            if len(parts) > 2 and parts[0] == 'module':
                return parts[1] + '.' + parts[2]
            return parts[0]
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in optimized_param_dict.values())
        optim_module_names = sorted(set(get_module_name(name) for name in optimized_param_dict.keys()))
        frozen_module_names = sorted(set(get_module_name(name) for name in set(all_param_dict.keys()) - set(optimized_param_dict.keys())))

        lr_info = ', '.join([
            f"{group.get('name', f'group_{idx}')}={group['lr']:.6g}"
            for idx, group in enumerate(optimizer.param_groups)
        ])
        print(f'Optimizer: AdamW, lr groups: {lr_info}, weight decay: {weight_decay}, betas: {betas}')
        print(f'Total parameters: {format_number(total_params)}, Trainable parameters: {format_number(trainable_params)}')
        print(f'Optimized parameters: {optim_module_names}')
        print(f'Frozen parameters: {frozen_module_names}')
        
    return optimizer, optimized_param_dict, all_param_dict

def create_lr_scheduler(optimizer, param_update_steps, warm_up_steps, scheduler_type='cosine'):
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    else:
        raise ValueError(f'Invalid scheduler type: {scheduler_type}')
    return scheduler



def find_checkpoints(load_path):
    if os.path.isdir(load_path):
        ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
        ckpt_names = sorted(ckpt_names, key=lambda x: x)
        ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
    else:
        if load_path.endswith(".pt"):
            ckpt_paths = [load_path]
        else:
            ckpt_paths = []
    return ckpt_paths



def auto_resume_job(
    load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state,
    reset_lr_scheduler=False,
):
    """
    Resume training from the latest checkpoint in the specified directory.
    Returns the fwdbwd_pass_step and param_update_step.

    Args:
        load_path: If dir, load the last checkpoint in the directory.
            O.w., assume it's a ckpt and load it.
        model: model to be loaded
        optimizer: optimizer to be loaded
        lr_scheduler: lr scheduler to be loaded
        reset_training_state: whether to reset the training state
        reset_lr_scheduler: whether to reset only the lr scheduler 
            (load optimizer but reset scheduler and step counters)

    Returns:
        optimizer, lr_scheduler, forward_pass_step, param_update_step

    """
    forward_pass_step = 0
    param_update_step = 0
    all_ckpt_paths = find_checkpoints(load_path)
    if len(all_ckpt_paths) == 0:
        print_rank0(f"No checkpoint found in {load_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step
    try:
        ckpt_path = all_ckpt_paths[-1]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except:
        traceback.print_exc()
        print_rank0(f"Failed to load {ckpt_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step

    # Load model weights
    if isinstance(model, DDP):
        status = model.module.load_state_dict(checkpoint['model'], strict=False)
    else:
        status = model.load_state_dict(checkpoint['model'], strict=False)
    print_rank0(f"Loaded model from {os.path.abspath(ckpt_path)}, the status is {status}")

    # resume training state
    if not reset_training_state:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if not reset_lr_scheduler:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                forward_pass_step = checkpoint["fwdbwd_pass_step"]
                param_update_step = checkpoint["param_update_step"]
                print_rank0(f"Resumed optimizer and lr_scheduler from {ckpt_path}")
            else:
                print_rank0(f"Resumed optimizer from {ckpt_path}")
        except:
            traceback.print_exc()
            print_rank0(f"Failed to load optimizer and lr_scheduler from {ckpt_path}")
    
    return optimizer, lr_scheduler, forward_pass_step, param_update_step


