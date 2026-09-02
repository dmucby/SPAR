import torch
from torch import Tensor
from jaxtyping import Float
from einops import reduce, rearrange
from skimage.metrics import structural_similarity
import functools
import os
from PIL import Image
from utils import data_utils
import numpy as np
from easydict import EasyDict as edict
import json
from rich import print
from torchmetrics import JaccardIndex, Accuracy

import warnings
# Suppress warnings for LPIPS loss loading
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum.*")

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, "batch"]:
    """
    Compute Peak Signal-to-Noise Ratio between ground truth and predicted images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        PSNR values for each image in the batch
    """
    ground_truth = torch.clamp(ground_truth, 0, 1)
    predicted = torch.clamp(predicted, 0, 1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * torch.log10(mse) 



@functools.lru_cache(maxsize=None)
def get_lpips_model(net_type="vgg", device="cuda"):
    from lpips import LPIPS
    return LPIPS(net=net_type).to(device)

@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    normalize: bool = True,
) -> Float[Tensor, "batch"]:
    """
    Compute Learned Perceptual Image Patch Similarity between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width]
        predicted: Images with shape [batch, channel, height, width]
        The value range is [0, 1] when we have set the normalize flag to True.
        It will be [-1, 1] when the normalize flag is set to False.
    Returns:
        LPIPS values for each image in the batch (lower is better)
    """

    _lpips_fn = get_lpips_model(device=predicted.device)
    batch_size = 10  # Process in batches to save memory
    values = [
        _lpips_fn(
            ground_truth[i : i + batch_size],
            predicted[i : i + batch_size],
            normalize=normalize,
        )
        for i in range(0, ground_truth.shape[0], batch_size)
    ]
    result = torch.cat(values, dim=0)
    # Only squeeze if there are multiple dimensions, but keep at least 1D
    if result.dim() > 1:
        result = result.squeeze()
    # Ensure result is at least 1D
    if result.dim() == 0:
        result = result.unsqueeze(0)
    return result



@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    """
    Compute Structural Similarity Index between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        SSIM values for each image in the batch (higher is better)
    """
    ssim_values= []
    
    for gt, pred in zip(ground_truth, predicted):
        # Move to CPU and convert to numpy
        gt_np = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        
        # Calculate SSIM
        ssim = structural_similarity(
            gt_np,
            pred_np,
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        ssim_values.append(ssim)
    
    # Convert back to tensor on the same device as input
    return torch.tensor(ssim_values, dtype=predicted.dtype, device=predicted.device)


@torch.no_grad()
def export_metrics(
    result: edict,
):  
    """
    Compute evaluation metrics (PSNR, LPIPS, SSIM) during training.
    
    Args:
        result: EasyDict with .target.image [b, v, c, h, w] and .render [b, v, c, h, w]
    
    Returns:
        dict with summed psnr, lpips, ssim and count for later averaging
    """
    target = result.target.image  # [b, v, c, h, w]
    prediction = result.render    # [b, v, c, h, w]
    target = rearrange(target, "b v c h w -> (b v) c h w")
    prediction = rearrange(prediction, "b v c h w -> (b v) c h w")

    target = target.to(torch.float32)
    prediction = prediction.to(torch.float32)
    
    count = target.size(0)

    psnr_values = compute_psnr(target, prediction)
    lpips_values = compute_lpips(target, prediction)
    ssim_values = compute_ssim(target, prediction)

    metrics = {
        "psnr": float(psnr_values.sum()),
        "lpips": float(lpips_values.sum()),
        "ssim": float(ssim_values.sum()),
        "count": count
    }
    return metrics


@torch.no_grad()
def compute_semantic_metrics(
    predicted: Float[Tensor, "batch height width"],
    target: Float[Tensor, "batch height width"],
    num_classes: int = 9,  # 8 semantic classes + 1 background/ignore class
    ignore_index: int = 0,
):
    """
    Compute semantic segmentation metrics (mIoU and Accuracy).
    
    Args:
        predicted: Predicted semantic labels [batch, height, width], values 0-8
        target: Ground truth semantic labels [batch, height, width], values 0-8
        num_classes: Total number of classes (including ignore class)
        ignore_index: Index to ignore in metric calculation (typically 0 for background)
        
    Returns:
        miou: Mean Intersection over Union (float)
        accuracy: Pixel accuracy (float)
    """
    device = predicted.device
    
    # Initialize metric computers
    jaccard = JaccardIndex(
        task='multiclass',
        num_classes=num_classes,
        ignore_index=ignore_index,
        average='macro'
    ).to(device)
    
    acc = Accuracy(
        task='multiclass',
        num_classes=num_classes,
        ignore_index=ignore_index,
        average='micro'
    ).to(device)
    
    # Ensure tensors are long type
    predicted = predicted.long()
    target = target.long()
    
    # Compute metrics
    miou = jaccard(predicted, target)
    accuracy = acc(predicted, target)
    
    return miou.item(), accuracy.item()



@torch.no_grad()
def export_results(
    result: edict,
    out_dir: str, 
    compute_metrics: bool = False
):
    """
    Save results including images and optional metrics and videos.
    
    Args:
        result: EasyDict containing input, target, and rendered images, and optionally video frames
        out_dir: Directory to save the evaluation results
        compute_metrics: Whether to compute and save metrics
    """
    os.makedirs(out_dir, exist_ok=True)
    
    input_data, target_data = result.input, result.target
    
    for batch_idx in range(input_data.image.size(0)):
        uid = input_data.index[batch_idx, 0, -1].item()
        scene_name = input_data.scene_name[batch_idx]
        sample_dir = os.path.join(out_dir, f"{uid:06d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Get target view indices
        target_indices = target_data.index[batch_idx, :, 0].cpu().numpy()
        
        # Save images
        _save_images(result, batch_idx, sample_dir)
        
        # Compute and save metrics if requested (requires rendered images)
        if compute_metrics and result.render is not None:
            # Collect semantic predictions and ground truth for ALL views (input + target)
            all_semantic_preds = []
            all_semantic_targets = []
            all_view_names = []
            
            # Get input (source) views semantic predictions and targets
            if hasattr(result, 'input_semantic_pred') and result.input_semantic_pred is not None:
                input_preds = result.input_semantic_pred[batch_idx]  # [v_input, h, w]
                if hasattr(input_data, 'labelmap') and input_data.labelmap is not None:
                    input_targets = input_data.labelmap[batch_idx]  # [v_input, h, w]
                    for v_idx in range(input_preds.shape[0]):
                        all_semantic_preds.append(input_preds[v_idx])
                        all_semantic_targets.append(input_targets[v_idx])
                        all_view_names.append(f"source_{v_idx+1}")
            
            # Get target view semantic predictions and targets
            if hasattr(result, 'target_semantic_pred') and result.target_semantic_pred is not None:
                target_preds = result.target_semantic_pred[batch_idx]  # [v_target, h, w]
                if hasattr(target_data, 'labelmap') and target_data.labelmap is not None:
                    target_targets = target_data.labelmap[batch_idx]  # [v_target, h, w]
                    for v_idx in range(target_preds.shape[0]):
                        all_semantic_preds.append(target_preds[v_idx])
                        all_semantic_targets.append(target_targets[v_idx])
                        all_view_names.append(f"target_{v_idx+1}")
            
            _save_metrics(
                target_data.image[batch_idx],
                result.render[batch_idx],
                target_indices,
                sample_dir,
                scene_name,
                all_semantic_preds=all_semantic_preds,
                all_semantic_targets=all_semantic_targets,
                all_view_names=all_view_names
            )
        
        # Save video if available
        if hasattr(result, "video_rendering"):
            _save_video(result.video_rendering[batch_idx], sample_dir)

def visualize_intermediate_results(out_dir, result):
    os.makedirs(out_dir, exist_ok=True)

    input, target = result.input, result.target

    if result.render is not None:
        target_image = target.image
        rendered_image = result.render
        b, v, _, h, w = rendered_image.size()
        rendered_image = rendered_image.reshape(b * v, -1, h, w)
        target_image = target_image.reshape(b * v, -1, h, w)
        visualized_image = torch.cat((target_image, rendered_image), dim=3).detach().cpu()
        visualized_image = rearrange(visualized_image, "(b v) c h (m w) -> (b h) (v m w) c", v=v, m=2)
        visualized_image = (visualized_image.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
        
        if hasattr(target, 'index') and target.index is not None:
            uids = [target.index[b, 0, -1].item() for b in range(target.index.size(0))]
            uid_based_filename = f"{uids[0]:08}_{uids[-1]:08}"
        else:
            uid_based_filename = "nouid"

        Image.fromarray(visualized_image).save(
            os.path.join(out_dir, f"supervision_{uid_based_filename}.jpg")
        )
        if hasattr(target, 'index') and target.index is not None:
            with open(os.path.join(out_dir, f"uids.txt"), "w") as f:
                uids = "_".join([f"{uid:08}" for uid in uids])
                f.write(uids)

    if hasattr(input, 'index') and input.index is not None:
        input_uids = [input.index[b, 0, -1].item() for b in range(input.index.size(0))]
        input_uid_based_filename = f"{input_uids[0]:08}_{input_uids[-1]:08}"
    else:
        input_uid_based_filename = "nouid"
    
    # Create a grid of input images
    b, v, c, h, w = input.image.size()
    input_images = input.image.reshape(b * v, c, h, w).detach().cpu()
    input_grid = rearrange(input_images, "(b v) c h w -> (b h) (v w) c", v=v)
    input_grid = (input_grid.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    
    # Save the input image grid
    Image.fromarray(input_grid).save(
        os.path.join(out_dir, f"input_{input_uid_based_filename}.jpg")
    )


def _save_images(result, batch_idx, out_dir):
    """Save visualization images."""
    # Save input image
    input_img = result.input.image[batch_idx]
    input_img = rearrange(input_img, "v c h w -> h (v w) c")
    input_img = (input_img.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(input_img).save(os.path.join(out_dir, "input.png"))

    # Skip GT vs prediction comparison if render is None
    if result.render is None:
        return

    # Save GT vs prediction side-by-side
    # Handle both [v,c,h,w] and [c,h,w] shapes
    target_img = result.target.image[batch_idx]
    render_img = result.render[batch_idx]
    
    # Ensure both are 4D [v,c,h,w]
    if target_img.dim() == 3:  # [c,h,w]
        target_img = target_img.unsqueeze(0)  # [1,c,h,w]
    if render_img.dim() == 3:  # [c,h,w]
        render_img = render_img.unsqueeze(0)  # [1,c,h,w]
    
    comparison = torch.cat((target_img, render_img), dim=2).detach().cpu()
    comparison = rearrange(comparison, "v c h w -> h (v w) c")
    comparison = (comparison.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(comparison).save(os.path.join(out_dir, "gt_vs_pred.png"))
    

def _save_metrics(target, prediction, view_indices, out_dir, scene_name, 
                 all_semantic_preds=None, all_semantic_targets=None, all_view_names=None):
    try:
        target = target.to(torch.float32)
        prediction = prediction.to(torch.float32)
        
        # Ensure both are 4D [v, c, h, w]
        if target.dim() == 3:  # [c, h, w]
            target = target.unsqueeze(0)
        if prediction.dim() == 3:  # [c, h, w]
            prediction = prediction.unsqueeze(0)
        
        # Ensure target and prediction have the same dimensions
        # This can happen when rendering resolution differs from target resolution
        if target.shape != prediction.shape:
            # Resize prediction to match target shape
            # target shape: [v, c, h, w]
            target_h, target_w = target.shape[-2:]
            pred_h, pred_w = prediction.shape[-2:]
            
            if (pred_h != target_h) or (pred_w != target_w):
                import torch.nn.functional as F
                prediction = F.interpolate(
                    prediction, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                )
        
        psnr_values = compute_psnr(target, prediction)
        lpips_values = compute_lpips(target, prediction)
        ssim_values = compute_ssim(target, prediction)
        
        # Compute semantic metrics for all views separately
        semantic_metrics_per_view = {}
        if all_semantic_preds is not None and all_semantic_targets is not None and all_view_names is not None:
            for sem_pred, sem_target, view_name in zip(all_semantic_preds, all_semantic_targets, all_view_names):
                try:
                    # Ensure both are long tensors and same shape
                    if sem_pred.dim() == 2:  # [h, w]
                        sem_pred = sem_pred.unsqueeze(0)
                    if sem_target.dim() == 2:  # [h, w]
                        sem_target = sem_target.unsqueeze(0)
                    
                    # Resize if needed
                    if sem_target.shape != sem_pred.shape:
                        sem_target_h, sem_target_w = sem_target.shape[-2:]
                        sem_pred_h, sem_pred_w = sem_pred.shape[-2:]
                        if (sem_pred_h != sem_target_h) or (sem_pred_w != sem_target_w):
                            import torch.nn.functional as F
                            sem_pred = F.interpolate(
                                sem_pred.unsqueeze(1).float(),
                                size=(sem_target_h, sem_target_w),
                                mode='nearest'
                            ).squeeze(1).long()
                    
                    miou, accuracy = compute_semantic_metrics(
                        sem_pred.long(),
                        sem_target.long()
                    )
                    semantic_metrics_per_view[view_name] = {
                        "miou": miou,
                        "accuracy": accuracy
                    }
                except Exception as e:
                    print(f"Warning: Failed to compute semantic metrics for {view_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    semantic_metrics_per_view[view_name] = {
                        "miou": float('nan'),
                        "accuracy": 0.0
                    }
    except Exception as e:
        print(f"Warning: Failed to compute metrics for {scene_name}: {e}")
        import traceback
        traceback.print_exc()
        return  # Skip saving metrics for this sample
    
    # Ensure values are at least 1D tensors (handle 0-dim tensors from squeeze)
    if psnr_values.dim() == 0:
        psnr_values = psnr_values.unsqueeze(0)
    if lpips_values.dim() == 0:
        lpips_values = lpips_values.unsqueeze(0)
    if ssim_values.dim() == 0:
        ssim_values = ssim_values.unsqueeze(0)

    # Build metrics dictionary
    metrics = {
        "summary": {
            "scene_name": scene_name,
            "psnr": float(psnr_values.mean()),
            "lpips": float(lpips_values.mean()),
            "ssim": float(ssim_values.mean())
        },
        "per_view": [],
        "semantic_per_view": {}  # Separate section for semantic metrics by view name
    }
    
    # Add RGB metrics per view
    for i, view_idx in enumerate(view_indices):
        # Convert tensor values to Python floats (psnr_values[i] is a scalar tensor)
        view_metrics = {
            "view": int(view_idx), 
            "psnr": float(psnr_values[i].item()), 
            "lpips": float(lpips_values[i].item()), 
            "ssim": float(ssim_values[i].item())
        }
        metrics["per_view"].append(view_metrics)
    
    # Add semantic metrics per view (source_1, source_2, target_1)
    if semantic_metrics_per_view:
        metrics["semantic_per_view"] = {
            view_name: {
                "miou": float(metrics_dict["miou"]) if not np.isnan(metrics_dict["miou"]) else None,
                "accuracy": float(metrics_dict["accuracy"])
            }
            for view_name, metrics_dict in semantic_metrics_per_view.items()
        }
        
        # Calculate merged metrics for source views (source_1 + source_2)
        # Find source view predictions and targets
        source_preds = []
        source_targets = []
        for i, view_name in enumerate(all_view_names):
            if view_name.startswith("source_"):
                source_preds.append(all_semantic_preds[i])
                source_targets.append(all_semantic_targets[i])
        
        # Compute merged source metrics
        if source_preds and source_targets:
            try:
                # Resize predictions to match targets if needed
                resized_source_preds = []
                for pred, target in zip(source_preds, source_targets):
                    if pred.shape != target.shape:
                        # Resize pred to match target
                        import torch.nn.functional as F
                        if pred.dim() == 2:
                            pred = pred.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
                        elif pred.dim() == 3:
                            pred = pred.unsqueeze(1)  # [n, 1, h, w]
                        
                        target_h, target_w = target.shape[-2:]
                        pred = F.interpolate(
                            pred.float(),
                            size=(target_h, target_w),
                            mode='nearest'
                        ).squeeze().long()
                    
                    resized_source_preds.append(pred)
                
                # Stack all source predictions and targets
                merged_source_pred = torch.stack(resized_source_preds, dim=0)  # [n_sources, h, w]
                merged_source_target = torch.stack(source_targets, dim=0)  # [n_sources, h, w]
                
                # Compute metrics on merged data
                source_miou, source_accuracy = compute_semantic_metrics(
                    merged_source_pred.long(),
                    merged_source_target.long()
                )
                
                metrics["summary"]["source_miou"] = float(source_miou)
                metrics["summary"]["source_accuracy"] = float(source_accuracy)
            except Exception as e:
                print(f"Warning: Failed to compute merged source metrics: {e}")
                import traceback
                traceback.print_exc()
        
        # Get target metrics separately
        target_mious = [m["miou"] for name, m in semantic_metrics_per_view.items() 
                       if name.startswith("target_") and not np.isnan(m["miou"])]
        target_accs = [m["accuracy"] for name, m in semantic_metrics_per_view.items() 
                      if name.startswith("target_")]
        
        if target_mious:
            metrics["summary"]["target_miou"] = float(np.mean(target_mious))
        if target_accs:
            metrics["summary"]["target_accuracy"] = float(np.mean(target_accs))
        
        # Calculate average semantic metrics across all views (for backward compatibility)
        all_mious = [m["miou"] for m in semantic_metrics_per_view.values() if not np.isnan(m["miou"])]
        all_accs = [m["accuracy"] for m in semantic_metrics_per_view.values()]
        
        if all_mious:
            metrics["summary"]["avg_miou"] = float(np.mean(all_mious))
        if all_accs:
            metrics["summary"]["avg_accuracy"] = float(np.mean(all_accs))
    
    # Save metrics to a single JSON file
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def _save_video(frames, out_dir):
    """
    Save video from rendered frames.
    Input frames should be in [v, c, h, w] format.
    """
    frames = np.ascontiguousarray(np.array(frames.cpu().to(torch.float32)))
    frames = rearrange(frames, "v c h w -> v h w c")
    data_utils.create_video_from_frames(
        frames, 
        f"{out_dir}/rendered_video.mp4", 
        framerate=30
    )


def summarize_evaluation(evaluation_folder):
    # Find and sort all valid subfolders
    subfolders = sorted(
        [
            os.path.join(evaluation_folder, dirname)
            for dirname in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, dirname))
        ],
        key=lambda x: (0, int(os.path.basename(x))) if os.path.basename(x).isdigit() else (1, os.path.basename(x))
    )

    metrics = {}
    valid_subfolders = []
    
    for subfolder in subfolders:
        json_path = os.path.join(subfolder, "metrics.json")
        if not os.path.exists(json_path):
            print(f"!!! Metrics file not found in {subfolder}, skipping...")
            continue
            
        valid_subfolders.append(subfolder)
        
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                # Extract summary metrics
                for metric_name, metric_value in data["summary"].items():
                    if metric_name == "scene_name":
                        continue
                    metrics.setdefault(metric_name, []).append(metric_value)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading metrics from {json_path}: {e}")

    if not valid_subfolders:
        print(f"No valid metrics files found in {evaluation_folder}")
        return

    csv_file = os.path.join(evaluation_folder, "summary.csv")
    with open(csv_file, "w") as f:
        header = ["Index"] + list(metrics.keys())
        f.write(",".join(header) + "\n")
        
        for i, subfolder in enumerate(valid_subfolders):
            basename = os.path.basename(subfolder)
            values = [str(metric_values[i]) for metric_values in metrics.values()]
            f.write(f"{basename},{','.join(values)}\n")
        
        f.write("\n")
        
        averages = [str(sum(values) / len(values)) for values in metrics.values()]
        f.write(f"average,{','.join(averages)}\n")
    
    print(f"Summary written to {csv_file}")
    print(f"Average: {','.join(averages)}")

    # export average metrics to a text file
    with open(os.path.join(evaluation_folder, "average_metrics.txt"), "w") as f:
        f.write(f"Average: {','.join(averages)}\n")
