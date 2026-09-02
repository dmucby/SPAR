# SPAR: Semantic-Photometric-Aware Reconstruction.
# Copyright (c) 2025 WildRayZer v4 + Semantic (New) implementation.
# Adapted for SPAR; upstream lineage is documented in THIRD_PARTY_NOTICES.md.
#
# Core design:
#   - A single forward path jointly runs CV-DRP and semantic reconstruction.
#   - Soft-masked positions receive a learned mask token plus 2D sin/cos PE.
#   - Loss = L_spar (reconstruction + regularization + BCE) + L_semantic.

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from PIL import Image
import traceback

from .loss import SemanticLossComputer
from .spar_loss import SPARLossComputer
from .transformer import QK_Norm_TransformerBlock, _init_weights_layerwise
from .transformer import init_weights as _init_weights
from .cv_drp import CrossViewDynamicRegionPredictor
from .dino_v3 import DINOv3FeatureExtractor, load_dinov3_from_local

from utils.spar_data_utils import SplitData
from utils.pe_utils import get_1d_sincos_pos_emb_from_grid, get_2d_sincos_pos_embed
from utils.pose_utils import rot6d2mat, quat2mat
from utils import camera_utils

from .rayzer import get_cam_se3, cam_info_to_plucker, PoseEstimator

class SPAR(nn.Module):
    """
    SPAR jointly reconstructs photometric and continuous semantic features
    while suppressing transient regions with CV-DRP.

    All datasets use the same CV-DRP and semantic reconstruction path.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.split_data = SplitData(config)

        # Get semantic feature dimension from config
        self.semantic_feat_dim = self.config.get("model", {}).get("lseg", {}).get("feature_dim", 512)

        # image tokenizer
        self.image_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.config.model.image_tokenizer.patch_size,
                pw=self.config.model.image_tokenizer.patch_size,
            ),
            nn.Linear(
                config.model.image_tokenizer.in_channels
                * (config.model.image_tokenizer.patch_size ** 2),
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.image_tokenizer.apply(_init_weights)

        # pe embedder
        self.use_pe_embedding_layer = config.model.get('input_with_pe', True)
        self.pe_embedder = (
            nn.Sequential(
                nn.Linear(config.model.transformer.d, config.model.transformer.d),
                nn.SiLU(),
                nn.Linear(config.model.transformer.d, config.model.transformer.d),
            )
            if self.use_pe_embedding_layer
            else nn.Identity()
        )
        self.pe_embedder.apply(_init_weights)

        # latent scene representation
        self.scene_code = nn.Parameter(
            torch.randn(config.model.scene_latent.length, config.model.transformer.d)
        )
        nn.init.trunc_normal_(self.scene_code, std=0.02)

        # pose tokens
        self.cam_code = nn.Parameter(
            torch.randn(
                self.config.model.pose_latent.get('length', 1),
                config.model.transformer.d,
            )
        )
        nn.init.trunc_normal_(self.cam_code, std=0.02)

        # temporal pe embedder
        self.temporal_pe_embedder = (
            nn.Sequential(
                nn.Linear(config.model.transformer.d, config.model.transformer.d),
                nn.SiLU(),
                nn.Linear(config.model.transformer.d, config.model.transformer.d),
            )
            if self.use_pe_embedding_layer
            else nn.Identity()
        )
        self.temporal_pe_embedder.apply(_init_weights)

        use_qk_norm = config.model.transformer.get("use_qk_norm", False)

        # transformer encoder (pose estimation)
        self.transformer_encoder = self._build_transformer_layers(
            config.model.transformer.encoder_n_layer,
            config.model.transformer.d,
            config.model.transformer.d_head,
            use_qk_norm,
            config.model.transformer.get("special_init", False),
            config.model.transformer.get("depth_init", False),
        )

        # transformer encoder_geom (scene encoding)
        self.transformer_encoder_geom = self._build_transformer_layers(
            config.model.transformer.encoder_geom_n_layer,
            config.model.transformer.d,
            config.model.transformer.d_head,
            use_qk_norm,
            config.model.transformer.get("special_init", False),
            config.model.transformer.get("depth_init", False),
        )

        self.decoder_ln = nn.LayerNorm(config.model.transformer.d, bias=False)

        # transformer decoder
        self.transformer_decoder = self._build_transformer_layers(
            config.model.transformer.decoder_n_layer,
            config.model.transformer.d,
            config.model.transformer.d_head,
            use_qk_norm,
            config.model.transformer.get("special_init", False),
            config.model.transformer.get("depth_init", False),
        )

        # pose predictor
        self.pose_predictor = PoseEstimator(config)

        # target pose tokenizers
        self.target_latent_h = config.model.target_image.height // config.model.target_image.patch_size
        self.target_latent_w = config.model.target_image.width // config.model.target_image.patch_size

        self.target_pose_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.config.model.target_image.patch_size,
                pw=self.config.model.target_image.patch_size,
            ),
            nn.Linear(
                config.model.target_image.in_channels
                * (config.model.target_image.patch_size ** 2),
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.target_pose_tokenizer.apply(_init_weights)

        self.target_pose_tokenizer2 = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.config.model.target_image.patch_size,
                pw=self.config.model.target_image.patch_size,
            ),
            nn.Linear(
                config.model.target_image.in_channels
                * (config.model.target_image.patch_size ** 2),
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.target_pose_tokenizer2.apply(_init_weights)

        # fuse mlp
        self.mlp_fuse = nn.Sequential(
            nn.LayerNorm(config.model.transformer.d * 2, bias=False),
            nn.Linear(config.model.transformer.d * 2, config.model.transformer.d, bias=True),
            nn.SiLU(),
            nn.Linear(config.model.transformer.d, config.model.transformer.d, bias=True),
        )
        self.mlp_fuse.apply(_init_weights)

        # output regressor (RGB)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(config.model.transformer.d, bias=False),
            nn.Linear(
                config.model.transformer.d,
                (config.model.target_image.patch_size ** 2) * 3,
                bias=False,
            ),
            nn.Sigmoid()
        )
        self.image_token_decoder.apply(_init_weights)

        # view type embedding
        # 0 = input view, 1 = target view
        self.view_type_embedding = nn.Embedding(2, config.model.transformer.d)
        nn.init.normal_(self.view_type_embedding.weight, std=0.02)

        # SPAR CV-DRP module

        # --- DINOv3 ViT-B/16 (frozen) ---
        dino_config = config.model.get('dino_v3', {})
        self.dino_v3_ckpt_path = dino_config.get(
            'checkpoint_path',
            './datasets/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
        )
        self.dino_v3_embed_dim = dino_config.get('embed_dim', 768)
        self._dino_v3 = None

        cv_drp_config = config.model.get('cv_drp', {})
        self.cv_drp = CrossViewDynamicRegionPredictor(
            patch_size=config.model.image_tokenizer.patch_size,
            mask_threshold=cv_drp_config.get('mask_threshold', 0.5),
            d_dino=self.dino_v3_embed_dim,
            d_image=config.model.transformer.d,
            d_ray=config.model.transformer.d,
            d_shared=config.model.transformer.d,
            vit_n_heads=cv_drp_config.get('vit_n_heads', 12),
            vit_n_layers=cv_drp_config.get('vit_n_layers', 4),
        )

        # Learned replacement for masked tokens, following the MAE convention.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.model.transformer.d))
        nn.init.normal_(self.mask_token, std=0.02)

        self.mask_ratio = config.model.get('mask_ratio', 0.10)
        self.use_clustered_mask = config.model.get('use_clustered_mask', True)

        input_masking_config = config.model.get('input_masking', {})
        self.input_masking_mode = input_masking_config.get(
            'mode',
            config.model.get('input_masking_mode', 'soft_with_pe'),
        )
        if self.input_masking_mode not in {'soft_with_pe', 'zero_out', 'learnable_indicator', 'none'}:
            raise ValueError(
                f"Unsupported input_masking_mode={self.input_masking_mode}. "
                "Expected one of {'soft_with_pe', 'zero_out', 'learnable_indicator', 'none'}."
            )

        self.input_mask_source = input_masking_config.get(
            'source',
            config.model.get('input_mask_source', 'dynamic'),
        )
        if self.input_mask_source not in {'dynamic', 'random'}:
            raise ValueError(
                f"Unsupported input_mask_source={self.input_mask_source}. "
                "Expected one of {'dynamic', 'random'}."
            )

        self.random_mask_ratio = input_masking_config.get(
            'random_mask_ratio',
            config.model.get('random_mask_ratio', self.mask_ratio),
        )

        mask_pipeline_config = config.model.get('dynamic_mask_pipeline', {})
        self.default_mask_source = mask_pipeline_config.get(
            'mask_source',
            config.model.get(
                'mask_source',
                config.training.get('mask_source', 'refined'),
            ),
        )
        if isinstance(self.default_mask_source, str):
            self.default_mask_source = self.default_mask_source.lower()
        if self.default_mask_source == 'sam2':
            self.default_mask_source = 'refined'
        if self.default_mask_source not in {'raw', 'refined', 'gt'}:
            raise ValueError(
                f"Unsupported default mask_source={self.default_mask_source}. "
                "Expected one of {'raw', 'refined', 'gt'}."
            )

        # Region-type embeddings used by learnable-indicator masking.
        d_model = config.model.transformer.d
        self.static_indicator = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dynamic_indicator = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.static_indicator, std=0.02)
        nn.init.normal_(self.dynamic_indicator, std=0.02)

        # LSeg semantic components
        self._init_semantic_modules(config)

        # Loss computers
        self.spar_loss_computer = SPARLossComputer(config)

        # Only cosine feature supervision is used from SemanticLossComputer.
        # Disable its unused RGB perceptual modules to avoid duplicate VGG copies.
        semantic_loss_config = copy.deepcopy(config)
        semantic_loss_config.training.lpips_loss_weight = 0.0
        semantic_loss_config.training.perceptual_loss_weight = 0.0
        self.semantic_loss_computer = SemanticLossComputer(semantic_loss_config)

        # Plucker-ray scale normalization across datasets
        plucker_norm_config = config.model.get('plucker_normalization', {})
        self.plucker_norm_mode = plucker_norm_config.get('mode', 'none')
        if self.plucker_norm_mode == 'learnable':
            self.log_plucker_scale = nn.Parameter(torch.zeros(1))
        elif self.plucker_norm_mode not in ('none', 'per_scene'):
            raise ValueError(f"Unknown plucker_normalization.mode: {self.plucker_norm_mode}")

        # config backup
        self.config_bk = copy.deepcopy(config)
        self.render_interpolate = config.training.get("render_interpolate", False)

        self.freeze_cv_drp_enabled = bool(config.training.get('freeze_cv_drp', False))

        # training settings
        if config.inference or config.get("evaluation", False):
            self.random_index = config.training.get('random_split', False)
        else:
            self.random_index = config.training.get('random_split', False)
        print('Use random index:', self.random_index)

        self.semantic_loss_weight = config.training.get('semantic_loss_weight', 0.1)
        pose_supervision = config.training.get('pose_supervision', {})
        self.pose_loss_weight = pose_supervision.get('weight', config.training.get('pose_loss_weight', 0.0))
        self.pose_translation_loss_weight = pose_supervision.get('translation_weight', 1.0)
        self.pose_rotation_loss_weight = pose_supervision.get('rotation_weight', 1.0)
        self.pose_focal_loss_weight = pose_supervision.get('focal_weight', 0.1)

        dynamic_loss_weighting = config.training.get('dynamic_loss_weighting', {})
        self.apply_dynamic_rgb_loss_weighting = dynamic_loss_weighting.get(
            'rgb',
            config.training.get('apply_dynamic_rgb_loss_weighting', True),
        )
        self.apply_dynamic_semantic_loss_weighting = dynamic_loss_weighting.get(
            'semantic',
            config.training.get('apply_dynamic_semantic_loss_weighting', True),
        )
        self.enable_semantic_feature_supervision = (
            self.use_lseg and self.semantic_loss_weight > 0.0
        )
        if self.use_lseg and not self.enable_semantic_feature_supervision:
            self._freeze_semantic_supervision_head()

        print('[SPAR] v4 + Semantic (unified forward) initialized')

    def _canonicalize_gt_c2w(self, gt_c2w: torch.Tensor) -> torch.Tensor:
        canonical = self.config.model.pose_latent.get('canonical', 'first')
        if canonical == 'first':
            cano_idx = 0
        elif canonical == 'middle':
            cano_idx = gt_c2w.shape[1] // 2
        else:
            raise NotImplementedError(f"Unsupported canonical mode: {canonical}")

        gt_canonical = gt_c2w[:, cano_idx]
        gt_canonical_inv = torch.linalg.inv(gt_canonical)
        return gt_canonical_inv.unsqueeze(1) @ gt_c2w

    def _normalize_gt_fxfycxcy(self, gt_fxfycxcy: torch.Tensor) -> torch.Tensor:
        gt_norm = gt_fxfycxcy.clone()
        gt_norm[..., 0] /= float(self.config.model.target_image.width)
        gt_norm[..., 1] /= float(self.config.model.target_image.height)
        gt_norm[..., 2] /= float(self.config.model.target_image.width)
        gt_norm[..., 3] /= float(self.config.model.target_image.height)
        return gt_norm

    def _compute_pose_supervision_loss(
        self,
        pred_c2w: torch.Tensor,
        pred_fxfycxcy: torch.Tensor,
        gt_c2w: torch.Tensor,
        gt_fxfycxcy: torch.Tensor,
    ) -> edict:
        gt_c2w_canonical = self._canonicalize_gt_c2w(gt_c2w)
        gt_fxfycxcy_norm = self._normalize_gt_fxfycxcy(gt_fxfycxcy)

        pred_t = pred_c2w[..., :3, 3]
        gt_t = gt_c2w_canonical[..., :3, 3]
        translation_loss = F.smooth_l1_loss(pred_t, gt_t)

        pred_r = pred_c2w[..., :3, :3]
        gt_r = gt_c2w_canonical[..., :3, :3]
        rel_r = pred_r @ gt_r.transpose(-1, -2)
        trace = rel_r.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0 + 1e-6, max=1.0)
        rotation_loss = torch.acos(cos_theta).mean()

        focal_loss = F.smooth_l1_loss(pred_fxfycxcy[..., :2], gt_fxfycxcy_norm[..., :2])

        total_loss = (
            self.pose_translation_loss_weight * translation_loss
            + self.pose_rotation_loss_weight * rotation_loss
            + self.pose_focal_loss_weight * focal_loss
        )
        return edict(
            total=total_loss,
            translation=translation_loss,
            rotation=rotation_loss,
            focal=focal_loss,
            gt_c2w_canonical=gt_c2w_canonical,
            gt_fxfycxcy_norm=gt_fxfycxcy_norm,
        )

    # ------------------------------------------------------------------ #
    #                   Semantic Module Initialization                     #
    # ------------------------------------------------------------------ #
    def _init_semantic_modules(self, config):
        d_model = config.model.transformer.d
        patch_size = config.model.target_image.patch_size
        patch_area = patch_size ** 2

        self._init_lseg()
        if not self.use_lseg:
            return

        self.semantic_pool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.semantic_proj = nn.Linear(self.semantic_feat_dim + 6, d_model)
        self.semantic_proj.apply(_init_weights)

        self.target_pose_tokenizer_semantic = nn.Sequential(
            Rearrange("b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                      ph=config.model.target_image.patch_size,
                      pw=config.model.target_image.patch_size),
            nn.Linear(config.model.target_image.in_channels * (config.model.target_image.patch_size ** 2),
                      d_model, bias=False),
        )
        self.target_pose_tokenizer_semantic.apply(_init_weights)

        self.d_semantic_hidden = 64
        output_dim_hidden = patch_area * self.d_semantic_hidden
        self.semantic_token_decoder = nn.Sequential(
            nn.LayerNorm(d_model, bias=False),
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 4, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 4, output_dim_hidden, bias=False),
        )
        self.semantic_token_decoder.apply(_init_weights)

        self.semantic_feature_expansion = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(self.d_semantic_hidden, self.semantic_feat_dim, kernel_size=1, stride=1),
        )
        self.semantic_feature_expansion.apply(_init_weights)

    def _init_lseg(self):
        lseg_config = self.config.get("model", {}).get("lseg", {})
        lseg_ckpt_path = lseg_config.get("checkpoint_path", None)
        if lseg_ckpt_path is None:
            print("Warning: LSeg checkpoint_path not provided. Semantic features disabled.")
            self.use_lseg = False
            self.lseg_model = None
            return
        from .lseg import LSegFeatureExtractor
        self.lseg_model = LSegFeatureExtractor.from_pretrained(lseg_ckpt_path, half_res=True)
        self.lseg_model.eval()
        for param in self.lseg_model.parameters():
            param.requires_grad = False
        self.use_lseg = True
        print(f"LSeg model loaded from {lseg_ckpt_path}")

    def _freeze_semantic_supervision_head(self):
        """Freeze params used only by semantic feature distillation loss."""
        for module_name in ('semantic_token_decoder', 'semantic_feature_expansion'):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------ #
    #                          LSeg helpers                                #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract_lseg_features(self, images, plucker_half=None, return_raw_features=False):
        """Extract frozen LSeg features, optionally concatenated with Plucker rays."""
        if self.lseg_model is None:
            return (None, None) if return_raw_features else None
        b, v, c, h, w = images.shape
        images_flat = images.reshape(b * v, c, h, w)
        lseg_features = self.lseg_model.extract_features(images_flat)
        if plucker_half is not None:
            lseg_with_pose = torch.cat([lseg_features, plucker_half], dim=1)
        else:
            lseg_with_pose = lseg_features
        features = self.semantic_pool(lseg_with_pose)
        lseg_token_feature = rearrange(features, '(b v) c h w -> b (v h w) c', b=b, v=v)
        del features
        del lseg_with_pose
        if return_raw_features:
            return lseg_token_feature, lseg_features
        del lseg_features
        return lseg_token_feature

    # ------------------------------------------------------------------ #
    #                     DINOv3 / CV-DRP / SAM2 helpers                       #
    # ------------------------------------------------------------------ #
    @property
    def dino_v3(self) -> DINOv3FeatureExtractor:
        """Lazily load the frozen DINOv3 feature extractor."""
        if self._dino_v3 is None:
            if os.path.exists(self.dino_v3_ckpt_path):
                dino_arch = self.config.model.get('dino_v3', {}).get('arch', 'vitb16')
                raw_model = load_dinov3_from_local(
                    self.dino_v3_ckpt_path,
                    arch=dino_arch,
                    img_size=self.config.model.image_tokenizer.image_size,
                )
                self._dino_v3 = DINOv3FeatureExtractor(raw_model)
            else:
                raise FileNotFoundError(
                    f"[SPAR] DINOv3 checkpoint not found at {self.dino_v3_ckpt_path}."
                )
        return self._dino_v3

    @torch.no_grad()
    def extract_dino_features(self, images: torch.Tensor) -> torch.Tensor:
        dino = self.dino_v3.to(images.device)
        return dino.extract_patch_features(images)

    def _build_transformer_layers(self, n_layers, d, d_head, use_qk_norm, special_init, depth_init):
        layers = [
            QK_Norm_TransformerBlock(d, d_head, use_qk_norm=use_qk_norm)
            for _ in range(n_layers)
        ]
        if special_init:
            for idx in range(len(layers)):
                if depth_init:
                    std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    std = 0.02 / (2 * n_layers) ** 0.5
                layers[idx].apply(lambda module: _init_weights_layerwise(module, std))
            layers = nn.ModuleList(layers)
        else:
            layers = nn.ModuleList(layers)
            layers.apply(_init_weights)
        return layers

    def freeze_dino(self):
        """Keep the DINOv3 feature backbone frozen and in evaluation mode."""
        if self._dino_v3 is not None:
            for p in self._dino_v3.parameters():
                p.requires_grad = False
            self._dino_v3.eval()

    def freeze_cv_drp(self):
        """Freeze CV-DRP for explicit ablations or staged fine-tuning."""
        self.freeze_cv_drp_enabled = True
        for p in self.cv_drp.parameters():
            p.requires_grad = False
        self.cv_drp.eval()
        print('[SPAR] CV-DRP frozen.')

    def unfreeze_cv_drp(self):
        """Enable end-to-end CV-DRP optimization."""
        self.freeze_cv_drp_enabled = False
        for p in self.cv_drp.parameters():
            p.requires_grad = True
        self.cv_drp.train(self.training)
        print('[SPAR] CV-DRP trainable (end-to-end).')

    def init_sam2(self, sam2_config: str, sam2_checkpoint: str, device: str = 'cuda'):
        """Initialize the optional SAM2 image predictor for mask refinement."""
        import sys
        _model_dir = os.path.dirname(os.path.abspath(__file__))
        if _model_dir not in sys.path:
            sys.path.insert(0, _model_dir)
        from .build_sam2_manual import build_sam2_manual
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        stem = os.path.splitext(os.path.basename(sam2_config))[0]
        sam2_model = build_sam2_manual(
            config_name=stem, ckpt_path=sam2_checkpoint, device=device, mode="eval",
        )
        self._sam2_predictor = SAM2ImagePredictor(sam2_model)
        print(f'[SPAR] SAM2 loaded from {sam2_checkpoint}')

    def train(self, mode=True):
        super().train(mode)
        self.spar_loss_computer.eval()
        self.semantic_loss_computer.eval()
        if self._dino_v3 is not None:
            self._dino_v3.eval()
        if self.freeze_cv_drp_enabled:
            self.cv_drp.eval()
        else:
            self.cv_drp.train(mode)
        if self.use_lseg and self.lseg_model is not None:
            self.lseg_model.eval()
        return self

    def get_overview(self):
        c = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        a = lambda m: sum(p.numel() for p in m.parameters())
        overview = edict(
            image_tokenizer=c(self.image_tokenizer),
            pe_embedder=c(self.pe_embedder),
            temporal_pe_embedder=c(self.temporal_pe_embedder),
            scene_code=self.scene_code.data.numel(),
            cam_code=self.cam_code.data.numel(),
            transformer_encoder=c(self.transformer_encoder),
            transformer_encoder_geom=c(self.transformer_encoder_geom),
            transformer_decoder=c(self.transformer_decoder),
            mlp_fuse=c(self.mlp_fuse),
            target_pose_tokenizer=c(self.target_pose_tokenizer),
            target_pose_tokenizer2=c(self.target_pose_tokenizer2),
            image_token_decoder=c(self.image_token_decoder),
            pose_predictor=c(self.pose_predictor),
            view_type_embedding=c(self.view_type_embedding),
            cv_drp_trainable=c(self.cv_drp),
            cv_drp_total=a(self.cv_drp),
        )
        if self._dino_v3 is not None:
            overview.dino_v3_total = a(self._dino_v3)
        if self.use_lseg:
            overview.semantic_proj = c(self.semantic_proj)
            overview.target_pose_tokenizer_semantic = c(self.target_pose_tokenizer_semantic)
            overview.semantic_token_decoder = c(self.semantic_token_decoder)
            overview.semantic_feature_expansion = c(self.semantic_feature_expansion)
        if self.plucker_norm_mode == 'learnable':
            overview.log_plucker_scale = self.log_plucker_scale.item()
        overview.plucker_norm_mode = self.plucker_norm_mode
        return overview

    # Plücker Ray Scale Normalization

    def _normalize_plucker_rays(self, plucker_rays: torch.Tensor) -> torch.Tensor:
        """Normalize Plucker-ray moments to reduce cross-dataset scale shifts."""
        if self.plucker_norm_mode == 'none':
            return plucker_rays

        B, V, C, H, W = plucker_rays.shape
        moment = plucker_rays[:, :, :3]  # [B, V, 3, H, W]

        if self.plucker_norm_mode == 'per_scene':
            # Normalize each scene's moment coordinates to approximately [-1, 1].
            scale = moment.abs().reshape(B, -1).max(dim=1)[0]  # [B]
            scale = scale.clamp(min=1e-6).reshape(B, 1, 1, 1, 1)
            plucker_rays = plucker_rays.clone()
            plucker_rays[:, :, :3] = moment / scale

        elif self.plucker_norm_mode == 'learnable':
            # Start from the pretrained scale and learn a positive divisor.
            scale = torch.exp(self.log_plucker_scale)
            plucker_rays = plucker_rays.clone()
            plucker_rays[:, :, :3] = moment / scale

        return plucker_rays

    # Pair-based CV-DRP helpers

    @staticmethod
    def _get_view_pairs(n_views: int, mode: str = 'input'):
        if mode == 'input':
            if n_views <= 1:
                return [(0, 0)]
            return [(i, i + 1) for i in range(n_views - 1)]
        elif mode == 'target':
            pairs = [(i, n_views - 1 - i) for i in range(n_views // 2)]
            if n_views % 2 == 1:
                mid = n_views // 2
                pairs.append((mid, max(0, mid - 1)))
            return pairs
        else:
            raise ValueError(f"Unknown pair mode: {mode}")

    def _run_cv_drp_paired(
        self,
        dino_features: torch.Tensor,
        img_tokens: torch.Tensor,
        ray_tokens: torch.Tensor,
        h_patches: int,
        w_patches: int,
        mode: str = 'input',
    ) -> dict:
        B, V, N, _ = dino_features.shape
        ps = self.config.model.image_tokenizer.patch_size
        H, W = h_patches * ps, w_patches * ps
        device = dino_features.device

        pairs = self._get_view_pairs(V, mode)
        logit_buckets = [[] for _ in range(V)]

        for (i, j) in pairs:
            pair_dino = torch.stack([dino_features[:, i], dino_features[:, j]], dim=1)
            pair_img  = torch.stack([img_tokens[:, i],    img_tokens[:, j]],    dim=1)
            pair_ray  = torch.stack([ray_tokens[:, i],    ray_tokens[:, j]],    dim=1)

            pair_res = self.cv_drp(
                dino_features=pair_dino,
                image_tokens=pair_img,
                ray_tokens=pair_ray,
                h_patches=h_patches,
                w_patches=w_patches,
            )
            logit_buckets[i].append(pair_res['logits'][:, 0:1])
            if i != j:
                logit_buckets[j].append(pair_res['logits'][:, 1:2])

        per_view_logits = []
        for v_idx in range(V):
            bucket = logit_buckets[v_idx]
            if len(bucket) == 0:
                per_view_logits.append(torch.zeros(B, 1, 1, H, W, device=device))
            elif len(bucket) == 1:
                per_view_logits.append(bucket[0])
            else:
                per_view_logits.append(torch.stack(bucket, dim=0).mean(dim=0))

        logits = torch.cat(per_view_logits, dim=1)
        pixel_mask = torch.sigmoid(logits)

        return {
            'logits': logits,
            'pixel_mask': pixel_mask,
        }

    # SAM2 Mask Refinement

    @torch.no_grad()
    def _sam2_refine_masks(
        self,
        images_01: torch.Tensor,
        pixel_mask: torch.Tensor,
        view_indices: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self, '_sam2_predictor') or self._sam2_predictor is None:
            return pixel_mask

        B, V_sub = pixel_mask.shape[:2]
        H, W = pixel_mask.shape[3], pixel_mask.shape[4]
        device = pixel_mask.device
        binary_mask = (pixel_mask > 0.5).float()

        refined_list = []
        for b_i in range(B):
            view_refined = []
            for v_j in range(V_sub):
                v_idx = view_indices[b_i, v_j].item()
                img_np = (images_01[b_i, v_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                mask_np = binary_mask[b_i, v_j, 0].cpu().numpy()

                if mask_np.sum() < 10 or mask_np.sum() > (H * W - 10):
                    view_refined.append(binary_mask[b_i, v_j])
                    continue

                try:
                    self._sam2_predictor.set_image(img_np)
                    mask_logits = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_logits = mask_logits * 20.0 - 10.0
                    mask_input_low_res = F.interpolate(
                        mask_logits, size=(256, 256),
                        mode='bilinear', align_corners=False,
                    ).squeeze(0).numpy()

                    masks_out, scores, _ = self._sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        mask_input=mask_input_low_res,
                        multimask_output=True,
                    )
                    best_idx = np.argmax(scores)
                    refined_np = masks_out[best_idx].astype(np.float32)
                    view_refined.append(
                        torch.from_numpy(refined_np).unsqueeze(0).to(device)
                    )
                except Exception:
                    view_refined.append(binary_mask[b_i, v_j])

            refined_list.append(torch.stack(view_refined, dim=0))

        return torch.stack(refined_list, dim=0)

    def _select_dynamic_masks_for_rendering(
        self,
        data: dict,
        input_idx: torch.Tensor,
        target_idx: torch.Tensor,
        batch_idx: torch.Tensor,
        raw_input_mask: torch.Tensor,
        raw_target_mask: torch.Tensor,
        refined_input_mask: torch.Tensor,
        refined_target_mask: torch.Tensor,
    ):
        """Select raw, refined, or ground-truth masks for reconstruction."""
        mask_source = data.get("mask_source", None)
        if mask_source is None:
            mask_source = getattr(self, "mask_source_override", None)
        if mask_source is None:
            mask_source = self.default_mask_source

        if isinstance(mask_source, str):
            mask_source = mask_source.lower()
        if mask_source == "sam2":
            mask_source = "refined"

        if mask_source == "raw":
            return raw_input_mask, raw_target_mask, "raw"

        if mask_source == "gt":
            if "binary_mask" not in data:
                raise KeyError("mask_source=gt requires data['binary_mask']")

            gt_masks = data["binary_mask"].float()
            gt_input_mask = gt_masks[batch_idx, input_idx].unsqueeze(2).to(raw_input_mask.dtype)
            gt_target_mask = gt_masks[batch_idx, target_idx].unsqueeze(2).to(raw_target_mask.dtype)
            return gt_input_mask, gt_target_mask, "gt"

        return refined_input_mask, refined_target_mask, "refined"

    def _sample_random_dynamic_mask_like(self, selected_input_mask: torch.Tensor) -> torch.Tensor:
        """Sample a patch-aligned dynamic mask for the random-mask baseline."""
        B, V_in, _, H, W = selected_input_mask.shape
        device = selected_input_mask.device
        dtype = selected_input_mask.dtype
        ps = self.config.model.image_tokenizer.patch_size
        h_patches = H // ps
        w_patches = W // ps

        random_patch_mask = (
            torch.rand(B, V_in, 1, h_patches, w_patches, device=device) < self.random_mask_ratio
        ).to(dtype)
        random_dynamic_mask = F.interpolate(
            random_patch_mask.reshape(B * V_in, 1, h_patches, w_patches),
            size=(H, W),
            mode='nearest',
        )
        return random_dynamic_mask.reshape(B, V_in, 1, H, W)

    def _select_input_dynamic_mask(self, selected_input_mask: torch.Tensor) -> torch.Tensor:
        """Select the dynamic mask used for input-token masking."""
        if self.input_mask_source == 'random':
            return self._sample_random_dynamic_mask_like(selected_input_mask)
        return selected_input_mask

    @staticmethod
    def _loss_weight_mask_or_none(
        selected_target_mask: torch.Tensor,
        enable_dynamic_weighting: bool,
    ) -> torch.Tensor:
        """Return an all-static mask when dynamic loss weighting is disabled."""
        if enable_dynamic_weighting:
            return selected_target_mask
        return torch.zeros_like(selected_target_mask)

    # Soft Masking with mask_token + 2D sincos PE
    # Soft-masking path inherited from the upstream camera-token implementation.

    def _apply_soft_masking_with_pe(
        self,
        img_tokens: torch.Tensor,
        static_prob_flat: torch.Tensor,
        v_input: int,
        h_patches: int,
        w_patches: int,
    ) -> torch.Tensor:
        """Replace dynamic evidence with a learned token and spatial encoding."""
        B, L, D = img_tokens.shape
        n_patches = h_patches * w_patches
        device = img_tokens.device

        token_mask = (1.0 - static_prob_flat).detach()  # [B, V_input * N]

        # Step 1: Soft masking — SPAR static-region weighting
        img_tokens_masked = img_tokens * static_prob_flat.detach().unsqueeze(-1)  # [B, L, D]

        spatial_pe = get_2d_sincos_pos_embed(
            embed_dim=D,
            grid_size=(h_patches, w_patches),
            device=device,
        ).to(img_tokens.dtype)  # [n_patches, D]

        spatial_pe = spatial_pe.unsqueeze(0).repeat(1, v_input, 1)

        positioned_mask_tokens = self.mask_token.expand(B, L, -1) + spatial_pe.expand(B, -1, -1)

        mask_weight = token_mask.unsqueeze(-1)  # [B, L, 1]
        img_tokens_masked = img_tokens_masked + positioned_mask_tokens * mask_weight

        return img_tokens_masked

    def _apply_learnable_indicator(
        self,
        img_tokens: torch.Tensor,
        static_prob_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Add a probability-weighted static or dynamic region embedding."""
        B, L, D = img_tokens.shape

        # static_prob: [B, L] → [B, L, 1]
        static_weight = static_prob_flat.detach().unsqueeze(-1)
        dynamic_weight = 1.0 - static_weight

        static_ind = self.static_indicator.expand(B, L, -1)   # [B, L, D]
        dynamic_ind = self.dynamic_indicator.expand(B, L, -1) # [B, L, D]

        indicator = static_weight * static_ind + dynamic_weight * dynamic_ind

        output_tokens = img_tokens + indicator

        return output_tokens

    def _apply_configurable_input_masking(
        self,
        img_tokens: torch.Tensor,
        static_prob_flat: torch.Tensor,
        v_input: int,
        h_patches: int,
        w_patches: int,
    ) -> torch.Tensor:
        """Apply the configured soft, zero-out, indicator, or no-mask policy."""
        static_weight = static_prob_flat.detach().unsqueeze(-1)

        if self.input_masking_mode == 'soft_with_pe':
            return self._apply_soft_masking_with_pe(
                img_tokens,
                static_prob_flat,
                v_input,
                h_patches,
                w_patches,
            )

        if self.input_masking_mode == 'zero_out':
            return img_tokens * static_weight

        if self.input_masking_mode == 'learnable_indicator':
            return self._apply_learnable_indicator(
                img_tokens,
                static_prob_flat,
            )

        if self.input_masking_mode == 'none':
            return img_tokens

        raise ValueError(
            f"Unsupported input_masking_mode={self.input_masking_mode}. "
            "Expected one of {'soft_with_pe', 'zero_out', 'learnable_indicator', 'none'}."
        )

    # Forward (CV-DRP + optional SAM2 + semantic branch)

    def forward(
        self,
        data,
        create_visual=False,
        render_video=False,
        iter=0,
        is_copypaste_augmented=False,
        paste_masks=None,
        pseudo_masks=None,
        is_dynamic_scene=False,
    ):
        """Run joint CV-DRP, photometric, and semantic reconstruction."""
        device = data['image'].device
        b = data['image'].shape[0]
        batch_idx = torch.arange(b).unsqueeze(1).to(device)

        # Step 0: Copy-Paste Augmentation
        paste_masks = None
        clean_images = data['image'].clone()

        if is_copypaste_augmented and hasattr(self, 'copy_paste_augmentor') \
                and self.copy_paste_augmentor is not None:
            with torch.no_grad():
                temp_data = {k: v for k, v in data.items()}
                aug_data, paste_masks = self.copy_paste_augmentor(temp_data)
                data['image'] = aug_data['image'].detach().clone()

        # Step 1: Split Data
        input, target, input_idx, target_idx = self.split_data(
            data, random_index=self.random_index
        )

        images_01 = data['image']                           # [B, V_all, C, H, W], [0,1]
        image_all = images_01 * 2.0 - 1.0                  # [-1, 1]
        b, v_input, c, h, w = input.image.shape
        v_all = image_all.shape[1]
        v_target = v_all - v_input
        input_idx = input_idx.to(device)
        target_idx = target_idx.to(device)
        batch_idx = torch.arange(b).unsqueeze(1).to(device)

        clean_target_images = clean_images[batch_idx, target_idx]  # [B, V_t, C, H, W]

        # Pose estimation always receives unmasked image tokens.
        img_tokens = self.image_tokenizer(image_all)        # [(b*v_all), n, d]
        _, n, d = img_tokens.shape
        if self.use_pe_embedding_layer:
            img_tokens = self.add_sptial_temporal_pe(img_tokens, b, v_all, h, w)
        img_tokens = rearrange(img_tokens, '(b v) n d -> b (v n) d', b=b, v=v_all)

        cam_tokens = self.get_camera_tokens(b, v_all)
        n_cam = cam_tokens.shape[1] // v_all
        assert n_cam == 1
        cam_tokens = rearrange(cam_tokens, 'b (v n) d -> b v n d', v=v_all)
        cam_tokens = rearrange(cam_tokens, 'b v n d -> b (v n) d')

        all_tokens = torch.cat([cam_tokens, img_tokens], dim=1)
        all_tokens = self.run_encoder(all_tokens)
        cam_tokens, _ = all_tokens.split([v_all * n_cam, v_all * n], dim=1)

        cam_tokens = rearrange(
            cam_tokens, 'b (v n) d -> (b v) n d', b=b, v=v_all, n=n_cam
        )[:, 0]
        cam_info = self.pose_predictor(cam_tokens, v_all)
        c2w, fxfycxcy = get_cam_se3(cam_info)
        pred_c2w_all = rearrange(c2w, '(b v) c d -> b v c d', b=b, v=v_all)
        pred_fxfycxcy_all = rearrange(fxfycxcy, '(b v) c -> b v c', b=b, v=v_all)
        normalized = True

        plucker_rays = cam_info_to_plucker(
            c2w, fxfycxcy, self.config.model.target_image, normalized=normalized
        )
        plucker_rays = rearrange(plucker_rays, '(b v) c h w -> b v c h w', b=b, v=v_all)

        plucker_rays = self._normalize_plucker_rays(plucker_rays)

        # Plücker ray tokenization
        plucker_emb_all = self.target_pose_tokenizer(plucker_rays)
        plucker_emb_all = rearrange(plucker_emb_all, '(b v) n d -> b v n d', b=b, v=v_all)

        plucker_emb_input = plucker_emb_all[batch_idx, input_idx]
        plucker_emb_input_flat = rearrange(plucker_emb_input, 'b v n d -> b (v n) d')

        plucker_emb_target = self.target_pose_tokenizer2(
            plucker_rays[batch_idx, target_idx]
        )

        # Pair-based CV-DRP predicts a mask for every view.
        ps = self.config.model.image_tokenizer.patch_size
        h_patches = h // ps
        w_patches = w // ps

        img_tokens_per_view = rearrange(
            img_tokens, 'b (v n_tok) d -> b v n_tok d', v=v_all
        )

        # DINOv3 patch features (frozen, no_grad)
        with torch.no_grad():
            dino_features_all = self.extract_dino_features(
                images_01.reshape(b * v_all, c, h, w)
            )
            dino_features_all = dino_features_all.reshape(b, v_all, n, -1)

        dino_input  = dino_features_all[batch_idx, input_idx]
        dino_target = dino_features_all[batch_idx, target_idx]
        img_tok_input_me  = img_tokens_per_view[batch_idx, input_idx]
        img_tok_target_me = img_tokens_per_view[batch_idx, target_idx]
        ray_tok_input  = plucker_emb_all[batch_idx, input_idx]
        ray_tok_target = plucker_emb_all[batch_idx, target_idx]

        # Pair-based CV-DRP prediction. DINO features above remain frozen,
        # while CV-DRP stays differentiable for end-to-end optimization.
        cv_drp_result_input = self._run_cv_drp_paired(
            dino_input, img_tok_input_me, ray_tok_input,
            h_patches, w_patches, mode='input',
        )
        cv_drp_result_target = self._run_cv_drp_paired(
            dino_target, img_tok_target_me, ray_tok_target,
            h_patches, w_patches, mode='target',
        )

        # SAM2 is bypassed unless a predictor was explicitly initialized.
        if getattr(self, "_sam2_predictor", None) is not None:
            refined_input_mask = self._sam2_refine_masks(
                images_01, cv_drp_result_input['pixel_mask'],
                input_idx, batch_idx,
            )
            refined_target_mask = self._sam2_refine_masks(
                images_01, cv_drp_result_target['pixel_mask'],
                target_idx, batch_idx,
            )
        else:
            refined_input_mask = cv_drp_result_input['pixel_mask']
            refined_target_mask = cv_drp_result_target['pixel_mask']

        selected_input_mask, selected_target_mask, selected_mask_source = (
            self._select_dynamic_masks_for_rendering(
                data=data,
                input_idx=input_idx,
                target_idx=target_idx,
                batch_idx=batch_idx,
                raw_input_mask=cv_drp_result_input['pixel_mask'],
                raw_target_mask=cv_drp_result_target['pixel_mask'],
                refined_input_mask=refined_input_mask,
                refined_target_mask=refined_target_mask,
            )
        )

        # Step 5: Soft Masking with mask_token + 2D sincos PE
        input_dynamic_mask = self._select_input_dynamic_mask(selected_input_mask)

        # static_prob = 1 − selected_dynamic_mask
        static_prob = 1.0 - input_dynamic_mask  # [B, v_in, 1, H, W]

        static_prob_patch = F.avg_pool2d(
            static_prob.reshape(b * v_input, 1, h, w),
            kernel_size=ps, stride=ps,
        )  # [B*v_in, 1, h_p, w_p]
        static_prob_flat = static_prob_patch.reshape(b, v_input * n)  # [B, v_in*N]

        img_tokens_input = img_tokens_per_view[batch_idx, input_idx]
        img_tokens_input = rearrange(img_tokens_input, 'b v n d -> b (v n) d')
        img_tokens_input = torch.cat([img_tokens_input, plucker_emb_input_flat], dim=-1)
        img_tokens_input = self.mlp_fuse(img_tokens_input)  # [b, v_in*n, d]

        # view type embedding
        img_tokens_input = img_tokens_input + self.view_type_embedding.weight[0]

        img_tokens_input = self._apply_configurable_input_masking(
            img_tokens_input, static_prob_flat,
            v_input, h_patches, w_patches,
        )

        # Extract, project, and mask LSeg semantic features.
        input_sem_tokens = None
        n_sem = 0
        if self.use_lseg:
            plucker_input_flat = rearrange(
                plucker_rays[batch_idx, input_idx], 'b v c h w -> (b v) c h w'
            )
            plucker_input_half = F.interpolate(
                plucker_input_flat, scale_factor=0.5, mode='bilinear', align_corners=False
            )
            lseg_token_feature = self.extract_lseg_features(
                input.image, plucker_half=plucker_input_half
            )
            input_sem_tokens = self.semantic_proj(lseg_token_feature)  # [B, V_in*h_sem*w_sem, D]

            input_sem_tokens = input_sem_tokens + self.view_type_embedding.weight[0]

            # LSeg half_res: [B*V, 512, H/2, W/2]
            # + plucker: [B*V, 518, H/2, W/2]
            # semantic_pool(kernel=8, stride=8): [B*V, 518, H/16, W/16]
            h_sem = h // 16
            w_sem = w // 16
            n_sem_per_view = h_sem * w_sem

            # static_prob: [B, v_in, 1, H, W] → [B, v_in, 1, h_sem, w_sem]
            static_prob_sem = F.avg_pool2d(
                static_prob.reshape(b * v_input, 1, h, w),
                kernel_size=16, stride=16,
            )  # [B*v_in, 1, h_sem, w_sem]
            static_prob_sem_flat = static_prob_sem.reshape(b, v_input * n_sem_per_view)  # [B, v_in*n_sem]

            input_sem_tokens = self._apply_configurable_input_masking(
                input_sem_tokens, static_prob_sem_flat,
                v_input, h_sem, w_sem,
            )

            n_sem = input_sem_tokens.shape[1]

        # Step 7: Scene Encoding (masked input tokens + masked semantic tokens)
        scene_tokens = self.scene_code.expand(b, -1, -1)
        n_scene = scene_tokens.shape[1]

        if input_sem_tokens is not None:
            all_tokens = torch.cat([scene_tokens, img_tokens_input, input_sem_tokens], dim=1)
        else:
            all_tokens = torch.cat([scene_tokens, img_tokens_input], dim=1)

        all_tokens = self.run_encoder_geom(all_tokens)
        scene_tokens, _ = all_tokens.split([n_scene, v_input * n + n_sem], dim=1)

        # Step 8: Rendering (RGB + Semantic)
        plucker_emb_target = plucker_emb_target + self.view_type_embedding.weight[1]

        plucker_emb_target_sem = None
        if self.enable_semantic_feature_supervision:
            plucker_emb_target_sem = self.target_pose_tokenizer_semantic(
                plucker_rays[batch_idx, target_idx]
            )
            plucker_emb_target_sem = plucker_emb_target_sem + self.view_type_embedding.weight[1]

        render_results = self._render_images_with_semantic(
            scene_tokens, plucker_emb_target, plucker_emb_target_sem
        )

        if create_visual and render_video:
            with torch.no_grad():
                c2w_target = rearrange(c2w, '(b v) c d -> b v c d', v=v_all)[batch_idx, target_idx]
                fxfycxcy_target = rearrange(fxfycxcy, '(b v) c -> b v c', v=v_all)[batch_idx, target_idx]
                c2w_target = rearrange(c2w_target, 'b v c d -> (b v) c d')
                fxfycxcy_target = rearrange(fxfycxcy_target, 'b v c -> (b v) c')
                vis_only_results = self.render_images_video(
                    scene_tokens, c2w_target, fxfycxcy_target, normalized=normalized,
                )

        # Step 9: Loss (L_spar + L_semantic)
        # --- SPAR Loss (recon + reg + bce) ---
        all_logits = torch.cat([
            cv_drp_result_input['logits'], cv_drp_result_target['logits']
        ], dim=1)
        all_pixel_mask = torch.cat([
            cv_drp_result_input['pixel_mask'], cv_drp_result_target['pixel_mask']
        ], dim=1)
        paste_masks_all = None
        if paste_masks is not None:
            pm_input  = paste_masks[batch_idx, input_idx]
            pm_target = paste_masks[batch_idx, target_idx]
            paste_masks_all = torch.cat([pm_input, pm_target], dim=1)

        loss_target_mask = self._loss_weight_mask_or_none(
            selected_target_mask,
            self.apply_dynamic_rgb_loss_weighting,
        )
        semantic_target_mask = self._loss_weight_mask_or_none(
            selected_target_mask,
            self.apply_dynamic_semantic_loss_weighting,
        )

        loss_metrics = self.spar_loss_computer.compute_spar_loss(
            rendered_images=render_results.rendered_images,
            target_images=target.image,
            target_logits=cv_drp_result_target['logits'],
            target_pixel_mask=loss_target_mask,
            all_logits=all_logits,
            is_copypaste_augmented=is_copypaste_augmented,
            paste_masks_all=paste_masks_all,
            all_pixel_mask=all_pixel_mask,
            clean_target_images=clean_target_images,
        )

        # Semantic supervision follows the same static-region weighting as RGB.
        if (
            self.use_lseg
            and self.semantic_loss_weight > 0.0
            and render_results.get('rendered_sem_features') is not None
        ):
            rendered_sem_for_loss = render_results.rendered_sem_features  # [B*V_t, D_sem, H/2, W/2]

            if is_copypaste_augmented and clean_target_images is not None:
                sem_gt_images = clean_target_images.reshape(-1, 3, h, w)
            else:
                sem_gt_images = target.image.reshape(-1, 3, h, w)

            with torch.no_grad():
                gt_lseg_features = self.lseg_model.extract_features(sem_gt_images)
                # gt_lseg_features: [B*V_t, D_sem, H/2, W/2]

            feature_loss_weight = (
                self.config.get("model", {})
                .get("lseg", {})
                .get("feature_loss_weight", 0.1)
            )
            if feature_loss_weight > 0.0:
                if is_copypaste_augmented and clean_target_images is not None:
                    # Copy-paste uses the clean image as full semantic supervision.
                    semantic_feature_loss = self.semantic_loss_computer.compute_semantic_feature_loss(
                        rendered_sem_for_loss,
                        gt_lseg_features,
                        semantic_weight_map=None,
                    )
                else:
                    # D-RE10K semantic loss is weighted by static probability.
                    # refined_target_mask: [B, V_t, 1, H, W], P(dynamic)
                    E_s_target = 1.0 - semantic_target_mask  # [B, V_t, 1, H, W]
                    E_s_flat = E_s_target.reshape(-1, 1, h, w)  # [B*V_t, 1, H, W]
                    _, _, h_sem_feat, w_sem_feat = rendered_sem_for_loss.shape
                    E_s_sem = F.interpolate(
                        E_s_flat, size=(h_sem_feat, w_sem_feat),
                        mode='bilinear', align_corners=False,
                    ).squeeze(1)  # [B*V_t, H/2, W/2]

                    semantic_feature_loss = self.semantic_loss_computer.compute_semantic_feature_loss(
                        rendered_sem_for_loss,
                        gt_lseg_features,
                        semantic_weight_map=E_s_sem,
                    )

                loss_metrics.semantic_feature_loss = semantic_feature_loss
                loss_metrics.loss = loss_metrics.loss + self.semantic_loss_weight * semantic_feature_loss
            else:
                loss_metrics.semantic_feature_loss = torch.tensor(0.0, device=device)
        else:
            loss_metrics.semantic_feature_loss = torch.tensor(0.0, device=device)

        pose_loss_stats = None
        if self.pose_loss_weight > 0.0:
            pose_loss_stats = self._compute_pose_supervision_loss(
                pred_c2w=pred_c2w_all,
                pred_fxfycxcy=pred_fxfycxcy_all,
                gt_c2w=data['c2w'],
                gt_fxfycxcy=data['fxfycxcy'],
            )
            loss_metrics.pose_loss = pose_loss_stats.total
            loss_metrics.pose_translation_loss = pose_loss_stats.translation
            loss_metrics.pose_rotation_loss = pose_loss_stats.rotation
            loss_metrics.pose_focal_loss = pose_loss_stats.focal
            loss_metrics.loss = loss_metrics.loss + self.pose_loss_weight * pose_loss_stats.total
        else:
            zero = torch.tensor(0.0, device=device)
            loss_metrics.pose_loss = zero
            loss_metrics.pose_translation_loss = zero
            loss_metrics.pose_rotation_loss = zero
            loss_metrics.pose_focal_loss = zero

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=render_results.rendered_images,
            render_semantic=render_results.get('rendered_sem_features'),
            c2w=pred_c2w_all,
            fxfycxcy=pred_fxfycxcy_all,
            input_idx=input_idx,
            target_idx=target_idx,
            # Raw and refined masks are exposed for evaluation and visualization.
            motion_mask_input_raw=cv_drp_result_input['pixel_mask'],
            motion_mask_target_raw=cv_drp_result_target['pixel_mask'],
            motion_mask_input_refined=refined_input_mask,
            motion_mask_target_refined=refined_target_mask,
            motion_mask_input_selected=selected_input_mask,
            motion_mask_target_selected=selected_target_mask,
            motion_mask_source=selected_mask_source,
            # Backward-compatible result aliases.
            motion_mask_input_soft=selected_input_mask,
            motion_mask_target_soft=selected_target_mask,
        )

        if pose_loss_stats is not None:
            result.gt_c2w_canonical = pose_loss_stats.gt_c2w_canonical
            result.gt_fxfycxcy = pose_loss_stats.gt_fxfycxcy_norm

        if create_visual and render_video:
            result.video_rendering = vis_only_results.rendered_images_video.detach()

        return result

    # Render helpers

    def _render_images_with_semantic(self, scene_tokens, target_tokens_rgb, target_tokens_sem=None):
        """Render RGB and semantic features from the shared scene tokens."""
        b, _, d = scene_tokens.shape
        bv = target_tokens_rgb.shape[0]
        v = bv // b
        scene_tokens = rearrange(
            scene_tokens.unsqueeze(1).repeat(1, v, 1, 1), 'b v n d -> (b v) n d'
        )
        n_target = target_tokens_rgb.shape[1]
        n_scene = scene_tokens.shape[1]

        if target_tokens_sem is not None:
            n_sem_target = target_tokens_sem.shape[1]
            all_tokens = torch.cat([target_tokens_rgb, target_tokens_sem, scene_tokens], dim=1)
        else:
            n_sem_target = 0
            all_tokens = torch.cat([target_tokens_rgb, scene_tokens], dim=1)

        all_tokens = self.decoder_ln(all_tokens)
        all_tokens = self.run_decoder(all_tokens)

        if n_sem_target > 0:
            target_rgb_tokens, target_sem_tokens, _ = all_tokens.split(
                [n_target, n_sem_target, n_scene], dim=1
            )
        else:
            target_rgb_tokens, _ = all_tokens.split([n_target, n_scene], dim=1)
            target_sem_tokens = None

        # RGB rendering
        ps = self.config.model.target_image.patch_size
        rendered_images = self.image_token_decoder(target_rgb_tokens)
        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v, h=self.target_latent_h, w=self.target_latent_w, p1=ps, p2=ps, c=3
        )
        render_results = edict(rendered_images=rendered_images)

        # Semantic rendering
        if (
            self.enable_semantic_feature_supervision
            and target_sem_tokens is not None
        ):
            decoded = self.semantic_token_decoder(target_sem_tokens)
            decoded = rearrange(
                decoded, "(b v) (h w) (p1 p2 c) -> (b v) c (h p1) (w p2)",
                v=v, h=self.target_latent_h, w=self.target_latent_w,
                p1=ps, p2=ps, c=self.d_semantic_hidden
            )
            render_results.rendered_sem_features = self.semantic_feature_expansion(decoded)

        return render_results

    # RayZer-compatible rendering helpers

    def get_camera_tokens(self, b, v):
        n, d = self.cam_code.shape[-2:]
        cam_tokens = rearrange(self.cam_code, 'n d -> 1 1 n d')
        cam_tokens = repeat(cam_tokens, '1 1 n d -> 1 v n d', v=v)
        cam_tokens = rearrange(cam_tokens, '1 v n d -> 1 (v n) d')
        img_indices = torch.arange(v).repeat_interleave(n).to(cam_tokens.device)
        temporal_pe = get_1d_sincos_pos_emb_from_grid(
            embed_dim=d, pos=img_indices, device=cam_tokens.device
        ).to(cam_tokens.dtype).reshape(1, v * n, d)
        temporal_pe = self.temporal_pe_embedder(temporal_pe)
        return (cam_tokens + temporal_pe).repeat(b, 1, 1)

    def add_sptial_temporal_pe(self, img_tokens, b, v, h_origin, w_origin):
        ps = self.config.model.image_tokenizer.patch_size
        nh, nw = h_origin // ps, w_origin // ps
        assert nh * nw == img_tokens.shape[1]
        bv, n, d = img_tokens.shape
        img_indices = torch.arange(v).repeat_interleave(n).unsqueeze(0).repeat(b, 1).reshape(-1).to(img_tokens.device)
        temporal_pe = get_1d_sincos_pos_emb_from_grid(
            embed_dim=d // 2, pos=img_indices, device=img_tokens.device
        ).to(img_tokens.dtype).reshape(b, v, n, d // 2)
        spatial_pe = get_2d_sincos_pos_embed(
            embed_dim=d // 2, grid_size=(nh, nw), device=img_tokens.device
        ).to(img_tokens.dtype).reshape(1, 1, n, d // 2).repeat(b, v, 1, 1)
        pe = self.pe_embedder(torch.cat([spatial_pe, temporal_pe], dim=-1).reshape(bv, n, d))
        return img_tokens + pe

    # Encoder/Decoder runners
    def run_layers_encoder(self, s, e):
        def f(x):
            for i in range(s, min(e, len(self.transformer_encoder))):
                x = self.transformer_encoder[i](x)
            return x
        return f

    def run_layers_encoder_geom(self, s, e):
        def f(x):
            for i in range(s, min(e, len(self.transformer_encoder_geom))):
                x = self.transformer_encoder_geom[i](x)
            return x
        return f

    def run_layers_decoder(self, s, e):
        def f(x):
            for i in range(s, min(e, len(self.transformer_decoder))):
                x = self.transformer_decoder[i](x)
            return x
        return f

    def run_encoder(self, tokens):
        ck = getattr(self.config.training, 'grad_checkpoint_every', 1)
        if ck <= 0:
            return self.run_layers_encoder(0, len(self.transformer_encoder))(tokens)
        for i in range(0, len(self.transformer_encoder), ck):
            tokens = torch.utils.checkpoint.checkpoint(
                self.run_layers_encoder(i, i + ck),
                tokens,
                use_reentrant=False,
            )
        return tokens

    def run_encoder_geom(self, tokens):
        ck = getattr(self.config.training, 'grad_checkpoint_every', 1)
        if ck <= 0:
            return self.run_layers_encoder_geom(0, len(self.transformer_encoder_geom))(tokens)
        for i in range(0, len(self.transformer_encoder_geom), ck):
            tokens = torch.utils.checkpoint.checkpoint(
                self.run_layers_encoder_geom(i, i + ck),
                tokens,
                use_reentrant=False,
            )
        return tokens

    def run_decoder(self, tokens):
        ck = getattr(self.config.training, 'grad_checkpoint_every', 1)
        if ck <= 0:
            return self.run_layers_decoder(0, len(self.transformer_decoder))(tokens)
        for i in range(0, len(self.transformer_decoder), ck):
            tokens = torch.utils.checkpoint.checkpoint(
                self.run_layers_decoder(i, i + ck),
                tokens,
                use_reentrant=False,
            )
        return tokens

    def render_images_video(self, scene_tokens_all, c2w_all, fxfycxcy_all, normalized=False):
        with torch.no_grad():
            scene_tokens_all = scene_tokens_all.detach()
            c2w_all, fxfycxcy_all = c2w_all.detach(), fxfycxcy_all.detach()
            b, _, d = scene_tokens_all.shape
            bv = c2w_all.shape[0]
            v = bv // b
            c2w_all = rearrange(c2w_all, '(b v) x y -> b v x y', v=v)
            fxfycxcy_all = rearrange(fxfycxcy_all, '(b v) x -> b v x', v=v)
            device = scene_tokens_all.device
            all_renderings = []
            num_frames = self.config.inference.render_video_config.num_frames
            traj_type = self.config.inference.render_video_config.traj_type
            for i in range(b):
                scene_tokens = scene_tokens_all[i]
                c2ws = c2w_all[i]
                fxfycxcy = fxfycxcy_all[i]
                if traj_type == "interpolate":
                    Ks = torch.zeros((c2ws.shape[0], 3, 3), device=device)
                    Ks[:, 0, 0], Ks[:, 1, 1] = fxfycxcy[:, 0], fxfycxcy[:, 1]
                    Ks[:, 0, 2], Ks[:, 1, 2] = fxfycxcy[:, 2], fxfycxcy[:, 3]
                    c2ws, Ks = camera_utils.get_interpolated_poses_many(
                        c2ws[:, :3, :4], Ks, num_frames, order_poses=False
                    )
                    frame_c2ws = torch.cat([
                        c2ws.to(device),
                        torch.tensor([[[0, 0, 0, 1]]], device=device).repeat(c2ws.shape[0], 1, 1)
                    ], dim=1)
                    frame_fxfycxcy = torch.zeros((c2ws.shape[0], 4), device=device)
                    frame_fxfycxcy[:, 0], frame_fxfycxcy[:, 1] = Ks[:, 0, 0], Ks[:, 1, 1]
                    frame_fxfycxcy[:, 2], frame_fxfycxcy[:, 3] = Ks[:, 0, 2], Ks[:, 1, 2]
                elif traj_type == "same":
                    frame_c2ws, frame_fxfycxcy = c2ws.clone(), fxfycxcy.clone()
                else:
                    raise NotImplementedError
                plucker = cam_info_to_plucker(
                    frame_c2ws, frame_fxfycxcy, self.config.model.target_image, normalized=normalized
                )
                # Match the Plucker-ray normalization used by forward().
                plucker = self._normalize_plucker_rays(plucker.unsqueeze(0)).squeeze(0)
                plk_emb = self.target_pose_tokenizer2(plucker.unsqueeze(0))
                v_r = plk_emb.shape[0]
                sc = scene_tokens.unsqueeze(0).repeat(v_r, 1, 1)
                at = torch.cat([plk_emb, sc], dim=1)
                nt, ns = plk_emb.shape[1], sc.shape[1]
                at = self.decoder_ln(at)
                at = self.run_decoder(at)
                target_tokens, _ = at.split([nt, ns], dim=1)
                rendered = self.image_token_decoder(target_tokens)
                ps = self.config.model.target_image.patch_size
                rendered = rearrange(
                    rendered, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                    v=v_r, h=self.target_latent_h, w=self.target_latent_w,
                    p1=ps, p2=ps, c=3
                ).squeeze(0)
                all_renderings.append(rendered)
            all_renderings = torch.stack(all_renderings)
        return edict(rendered_images_video=all_renderings)

    @torch.no_grad()
    def load_ckpt(self, load_path, strict=False):
        if os.path.isdir(load_path):
            ckpt_names = sorted([f for f in os.listdir(load_path) if f.endswith(".pt")])
            ckpt_paths = [os.path.join(load_path, n) for n in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        status = self.load_compatible_state_dict(checkpoint["model"], strict=strict)
        print(f"[SPAR] Loaded from {ckpt_paths[-1]}")
        if status.missing_keys:
            print(f"  Missing: {status.missing_keys[:15]}...")
        if status.unexpected_keys:
            print(f"  Unexpected: {status.unexpected_keys[:15]}...")
        return 0

    @staticmethod
    def _migrate_legacy_checkpoint_keys(state_dict):
        """Map pre-release CV-DRP parameter names to the public module name."""
        migrated = state_dict.__class__()
        for key, value in state_dict.items():
            if key.startswith('motion_estimator.'):
                key = 'cv_drp.' + key[len('motion_estimator.'):]
            migrated[key] = value
        if hasattr(state_dict, '_metadata'):
            migrated._metadata = {
                ('cv_drp' + key[len('motion_estimator'):]
                 if key.startswith('motion_estimator') else key): value
                for key, value in state_dict._metadata.items()
            }
        return migrated

    def load_compatible_state_dict(self, state_dict, strict=False):
        """Load public or pre-release checkpoints without changing predictions."""
        state_dict = self._migrate_legacy_checkpoint_keys(state_dict)
        return self.load_state_dict(state_dict, strict=strict)
