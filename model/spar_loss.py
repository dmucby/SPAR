# SPAR joint dynamic-region-aware loss.
# Copyright (c) 2025 WildRayZer implementation.
# Adapted for SPAR; see THIRD_PARTY_NOTICES.md.
#
# The loss combines static-region reconstruction, CV-DRP regularization,
# copy-paste mask supervision, perceptual loss, and optional LPIPS.

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from typing import Optional


class SPARLossComputer(nn.Module):
    """Dynamic-region-aware joint loss used to train SPAR."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        spar_config = config.training.get('spar_loss', {})

        self.l2_loss_weight = config.training.get('l2_loss_weight', 1.0)
        self.perceptual_loss_weight = config.training.get('perceptual_loss_weight', 0.2)
        self.lpips_loss_weight = config.training.get('lpips_loss_weight', 0.0)

        # SPAR joint loss weights
        self.reg_weight = spar_config.get('mask_regularization_weight', 0.01)   # λ_reg
        self.bce_weight = spar_config.get('mask_bce_weight', 1.0)               # λ_bce

        self._perceptual_loss = None
        self._lpips_loss = None

    # Lazy perception modules

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    @property
    def perceptual_loss_module(self):
        if self._perceptual_loss is None and self.perceptual_loss_weight > 0:
            from .loss import PerceptualLoss
            device = self._get_device()
            self._perceptual_loss = PerceptualLoss(device=device)
            self._perceptual_loss.eval()
            for p in self._perceptual_loss.parameters():
                p.requires_grad = False
        return self._perceptual_loss

    @property
    def lpips_loss_module(self):
        if self._lpips_loss is None and self.lpips_loss_weight > 0:
            import lpips
            device = self._get_device()
            self._lpips_loss = lpips.LPIPS(net="vgg").to(device)
            self._lpips_loss.eval()
            for p in self._lpips_loss.parameters():
                p.requires_grad = False
        return self._lpips_loss

    # SPAR joint loss

    def compute_spar_loss(
        self,
        rendered_images: torch.Tensor,           # [B, V_t, 3, H, W]
        target_images: torch.Tensor,
        target_logits: torch.Tensor,             # [B, V_t, 1, H, W]
        target_pixel_mask: torch.Tensor,         # [B, V_t, 1, H, W]  P(dynamic)
        all_logits: torch.Tensor,                # [B, V_all, 1, H, W]
        is_copypaste_augmented: bool = False,
        paste_masks_all: Optional[torch.Tensor] = None,  # [B, V_all, 1, H, W]
        all_pixel_mask: Optional[torch.Tensor] = None,   # [B, V_all, 1, H, W]
        clean_target_images: Optional[torch.Tensor] = None,
    ) -> edict:
        """Compute reconstruction, perceptual, and dynamic-region losses."""
        B, V_t, C, H, W = rendered_images.shape
        device = rendered_images.device
        render_flat = rendered_images.reshape(B * V_t, C, H, W)

        if is_copypaste_augmented and clean_target_images is not None:
            recon_gt = clean_target_images
            target_flat = recon_gt.reshape(B * V_t, C, H, W)
            L_recon = F.mse_loss(render_flat, target_flat)
            perceptual_loss = torch.tensor(0.0, device=device)
            if self.perceptual_loss_weight > 0 and self.perceptual_loss_module is not None:
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    perceptual_loss = self.perceptual_loss_module(
                        render_flat.float(), target_flat.float()
                    )
        else:
            recon_gt = target_images
            target_flat = recon_gt.reshape(B * V_t, C, H, W)
            E_s_target = 1.0 - target_pixel_mask
            E_s_flat = E_s_target.reshape(B * V_t, 1, H, W)# [B, V_t, 1, H, W]
            pixel_mse = (render_flat - target_flat) ** 2        # [B, V_t, 3, H, W]
            weight_sum = E_s_flat.sum() + 1e-8
            L_recon = (pixel_mse * E_s_flat).sum() / (weight_sum * C)

            perceptual_loss = torch.tensor(0.0, device=device)
            if self.perceptual_loss_weight > 0 and self.perceptual_loss_module is not None:
                render_flat = rendered_images.reshape(B * V_t, C, H, W)
                target_flat = recon_gt.reshape(B * V_t, C, H, W)
                E_s_flat = E_s_target.reshape(B * V_t, 1, H, W)
                target_masked = target_flat * E_s_flat + render_flat.detach() * (1.0 - E_s_flat)
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    perceptual_loss = self.perceptual_loss_module(
                        render_flat.float(), target_masked.float()
                    )

        # LPIPS loss
        lpips_loss = torch.tensor(0.0, device=device)
        if self.lpips_loss_weight > 0 and self.lpips_loss_module is not None:
            render_flat = rendered_images.reshape(B * V_t, C, H, W)
            gt_flat = recon_gt.reshape(B * V_t, C, H, W)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                lpips_loss = self.lpips_loss_module(
                    render_flat.float() * 2.0 - 1.0,
                    gt_flat.float() * 2.0 - 1.0,
                ).mean()

        # L_reg:  Explainability Regularization
        #   Penalize degenerate all-dynamic predictions while training CV-DRP.
        L_reg = torch.tensor(0.0, device=device)
        if self.reg_weight > 0 and all_logits is not None:
            B_all, V_all = all_logits.shape[:2]
            logits_flat = all_logits.reshape(B_all * V_all, 1, H, W)
            target_reg = torch.zeros_like(logits_flat)
            L_reg = F.binary_cross_entropy_with_logits(logits_flat, target_reg)

        #   Supervise CV-DRP on synthetically pasted objects.
        L_bce = torch.tensor(0.0, device=device)

        if self.bce_weight > 0 and is_copypaste_augmented \
                and paste_masks_all is not None and all_logits is not None:
            B_all_bce, V_all_bce = all_logits.shape[:2]
            has_paste = (
                paste_masks_all.reshape(B_all_bce, V_all_bce, -1).sum(dim=-1) > 0
            )  # [B, V_all]  bool

            if has_paste.any():
                pred_logits = all_logits[has_paste]           # [N_aug, 1, H, W]
                gt_paste    = paste_masks_all[has_paste]      # [N_aug, 1, H, W]
                L_bce = F.binary_cross_entropy_with_logits(
                    pred_logits.float(), gt_paste.float(),
                )

        # Total Loss
        total_loss = (
            self.l2_loss_weight * L_recon
            + self.perceptual_loss_weight * perceptual_loss
            + self.lpips_loss_weight * lpips_loss
            + self.reg_weight * L_reg
            + self.bce_weight * L_bce
        )

        with torch.no_grad():
            # Full-image PSNR
            full_mse = F.mse_loss(render_flat, target_flat)
            psnr_full = -10.0 * torch.log10(full_mse.clamp(min=1e-10))

            psnr_static = psnr_full
            psnr_dynamic = torch.tensor(0.0, device=device)

            if not is_copypaste_augmented and target_pixel_mask is not None:
                E_s_flat_det = E_s_flat.detach()
                static_region = (E_s_flat_det > 0.5)

                if static_region.any():
                    static_mse = (pixel_mse.detach() * static_region.float()).sum() / \
                                 (static_region.float().sum() * C + 1e-8)
                    psnr_static = -10.0 * torch.log10(static_mse.clamp(min=1e-10))

                dynamic_region = ~static_region
                if dynamic_region.any():
                    dynamic_mse = (pixel_mse.detach() * dynamic_region.float()).sum() / \
                                  (dynamic_region.float().sum() * C + 1e-8)
                    psnr_dynamic = -10.0 * torch.log10(dynamic_mse.clamp(min=1e-10))

        return edict(
            loss=total_loss,
            L_recon=L_recon,
            L_reg=L_reg,
            L_bce=L_bce,
            psnr=psnr_full,
            psnr_static=psnr_static,
            psnr_dynamic=psnr_dynamic,
            l2_loss=L_recon,
            L_vs=L_recon,
            recon_loss=L_recon,
            mask_bce_loss=L_bce,
            perceptual_loss=perceptual_loss,
            lpips_loss=lpips_loss,
        )


    @staticmethod
    def _compute_psnr_per_sample(
        rendered: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        N = rendered.shape[0]
        mse = F.mse_loss(rendered, target, reduction='none')
        mse = mse.reshape(N, -1).mean(dim=1)
        return -10.0 * torch.log10(mse.clamp(min=1e-10))

    @staticmethod
    def _compute_iou(
        pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6,
    ) -> torch.Tensor:
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)
