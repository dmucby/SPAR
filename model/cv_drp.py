"""Cross-View Dynamic Region Predictor (CV-DRP)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.transformer import QK_Norm_TransformerBlock, init_weights

class FusionMLP(nn.Module):
    """Fuse concatenated semantic, appearance, and geometric tokens."""

    def __init__(self, d_in: int, d_out: int, hidden_ratio: float = 2.0):
        super().__init__()
        d_hidden = int(d_out * hidden_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)

class DPTDecoder(nn.Module):
    """Upsample patch-level features into a per-pixel dynamic-region logit map."""

    def __init__(self, d_model: int = 768, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        channels = [256, 128, 64, 32]

        self.project = nn.Sequential(
            nn.Conv2d(d_model, channels[0], 1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )

        self.up_blocks = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ))
            in_ch = out_ch

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(channels[-1], 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )

        self.head = nn.Conv2d(16, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # A low initial dynamic probability avoids an all-dynamic initialization.
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, -4.6)

    def forward(self, tokens: torch.Tensor, h: int, w: int):
        """
        Args:
            tokens: [N, h*w, d_model]
            h, w: patch grid dimensions

        Returns:
            logits: [N, 1, H, W] raw logits (no sigmoid)
        """
        N = tokens.shape[0]
        x = tokens.transpose(1, 2).reshape(N, -1, h, w)  # [N, d_model, h, w]
        x = self.project(x)

        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.final_up(x)
        logits = self.head(x)  # [N, 1, H, W]
        return logits


class CVDRPFusion(nn.Module):
    """Fuse DINOv3, photometric, and Plucker-ray tokens across views."""

    def __init__(
        self,
        d_dino: int = 768,
        d_image: int = 768,
        d_ray: int = 768,
        d_shared: int = 768,
        n_heads: int = 12,
        n_transformer_layers: int = 4,
        patch_size: int = 16,
        mask_threshold: float = 0.5,
    ):
        super().__init__()
        self.d_shared = d_shared
        self.patch_size = patch_size
        self.mask_threshold = mask_threshold

        self.dino_proj = nn.Sequential(
            nn.LayerNorm(d_dino),
            nn.Linear(d_dino, d_shared),
        )
        self.image_proj = nn.Sequential(
            nn.LayerNorm(d_image),
            nn.Linear(d_image, d_shared),
        )
        self.ray_proj = nn.Sequential(
            nn.LayerNorm(d_ray),
            nn.Linear(d_ray, d_shared),
        )

        self.fusion_mlp = FusionMLP(
            d_in=3 * d_shared,
            d_out=d_shared,
            hidden_ratio=2.0,
        )

        head_dim = d_shared // n_heads
        self.transformer_layers = nn.ModuleList([
            QK_Norm_TransformerBlock(
                dim=d_shared,
                head_dim=head_dim,
                use_qk_norm=True,
            )
            for _ in range(n_transformer_layers)
        ])

        self.dpt_decoder = DPTDecoder(d_model=d_shared, patch_size=patch_size)

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if 'dpt_decoder' in name:
                continue
            module.apply(init_weights)

    def forward(
        self,
        dino_features: torch.Tensor,
        image_tokens: torch.Tensor,
        ray_tokens: torch.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> dict:
        """Predict per-pixel dynamic-region logits and masks for each view."""
        B, V, N, _ = dino_features.shape
        H = h_patches * self.patch_size
        W = w_patches * self.patch_size

        fused = torch.cat([dino_features, image_tokens, ray_tokens], dim=-1)  # [B, V, N, 3*d]
        fused = self.fusion_mlp(fused)  # [B, V, N, d_shared]

        fused_flat = rearrange(fused, 'b v n d -> b (v n) d')  # [B, V*N, d_shared]

        for layer in self.transformer_layers:
            fused_flat = layer(fused_flat)

        fused_per_view = rearrange(fused_flat, 'b (v n) d -> (b v) n d', v=V, n=N)

        logits = self.dpt_decoder(fused_per_view, h_patches, w_patches)  # [B*V, 1, H, W]
        logits = logits.reshape(B, V, 1, H, W)

        pixel_mask = torch.sigmoid(logits)
        binary_mask = (pixel_mask > self.mask_threshold).float()

        patch_mask = F.avg_pool2d(
            binary_mask.reshape(B * V, 1, H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )  # [B*V, 1, h, w]
        patch_mask = (patch_mask.reshape(B, V, -1) > 0.5).float()  # [B, V, N]

        return {
            'logits': logits,
            'pixel_mask': pixel_mask,
            'binary_mask': binary_mask,
            'patch_mask': patch_mask,
        }


class CrossViewDynamicRegionPredictor(nn.Module):
    """Public CV-DRP interface used by SPAR."""

    def __init__(
        self,
        patch_size: int = 16,
        mask_threshold: float = 0.5,
        d_dino: int = 768,
        d_image: int = 768,
        d_ray: int = 768,
        d_shared: int = 768,
        vit_n_heads: int = 12,
        vit_n_layers: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_threshold = mask_threshold
        # Keep this submodule name stable for released checkpoint compatibility.
        self.estimator = CVDRPFusion(
            d_dino=d_dino,
            d_image=d_image,
            d_ray=d_ray,
            d_shared=d_shared,
            n_heads=vit_n_heads,
            n_transformer_layers=vit_n_layers,
            patch_size=patch_size,
            mask_threshold=mask_threshold,
        )

    def forward(
        self,
        dino_features: torch.Tensor,
        image_tokens: torch.Tensor,
        ray_tokens: torch.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> dict:
        """Run cross-view dynamic-region prediction."""
        return self.estimator(
            dino_features=dino_features,
            image_tokens=image_tokens,
            ray_tokens=ray_tokens,
            h_patches=h_patches,
            w_patches=w_patches,
        )

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
