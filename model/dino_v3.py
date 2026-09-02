"""Thin wrapper around the vendored DINOv3 backbone."""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_model_dir = os.path.dirname(os.path.abspath(__file__))
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)

from dinov3.hub.backbones import dinov3_vitb16, dinov3_vitl16, dinov3_vits16


def load_dinov3_from_local(
    checkpoint_path: str,
    arch: str = "vitb16",
    img_size: int = 256,
    device: str = "cpu",
):
    """Load a frozen DINOv3 backbone from a local checkpoint."""
    print(f"[DINOv3] Loading {arch} from {checkpoint_path}...")

    arch_fn = {
        "vitb16": dinov3_vitb16,
        "vitl16": dinov3_vitl16,
        "vits16": dinov3_vits16,
    }

    if arch not in arch_fn:
        raise ValueError(f"Unsupported arch: {arch}. Choose from {list(arch_fn.keys())}")

    model = arch_fn[arch](
        pretrained=True,
        weights=checkpoint_path,
    )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[DINOv3] Loaded successfully. Params: {n_params:.1f}M, "
          f"embed_dim={model.embed_dim}, patch_size={model.patch_size}")

    return model.to(device)


class DINOv3FeatureExtractor(nn.Module):
    """Expose normalized patch features required by CV-DRP."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embed_dim = model.embed_dim
        self.patch_size = model.patch_size

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """ImageNet normalize + resize to patch-aligned."""
        images = (images - self.mean) / self.std
        _, _, H, W = images.shape
        new_H = (H // self.patch_size) * self.patch_size
        new_W = (W // self.patch_size) * self.patch_size
        if new_H != H or new_W != W:
            images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=False)
        return images

    @torch.no_grad()
    def forward_features(self, images: torch.Tensor):
        """Return the DINOv3 feature dictionary for images in the [0, 1] range."""
        x = self._preprocess(images)
        _, _, H, W = x.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size

        out = self.model.forward_features(x)
        out['h_patches'] = h_p
        out['w_patches'] = w_p
        out['patch_tokens'] = out['x_norm_patchtokens']
        out['cls_token'] = out['x_norm_clstoken']
        return out

    @torch.no_grad()
    def extract_patch_features(self, images: torch.Tensor, max_batch: int = 8):
        """Extract patch tokens in bounded-size batches."""
        all_feats = []
        for i in range(0, images.shape[0], max_batch):
            batch = images[i:i + max_batch]
            out = self.forward_features(batch)
            all_feats.append(out['x_norm_patchtokens'])
        return torch.cat(all_feats, dim=0)

    @torch.no_grad()
    def extract_spatial_features(self, images: torch.Tensor, max_batch: int = 8):
        """Extract patch tokens and restore their spatial layout."""
        all_feats = []
        for i in range(0, images.shape[0], max_batch):
            batch = images[i:i + max_batch]
            out = self.forward_features(batch)
            bs = batch.shape[0]
            h_p, w_p = out['h_patches'], out['w_patches']
            spatial = out['x_norm_patchtokens'].reshape(bs, h_p, w_p, -1).permute(0, 3, 1, 2)
            all_feats.append(spatial)
        return torch.cat(all_feats, dim=0)
