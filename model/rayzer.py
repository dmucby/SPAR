# Copyright (c) 2025 Hanwen Jiang. Created for the RayZer project.
# Enhanced with LSeg-based semantic segmentation (integrated from LVSM_scene_decoder_only_semantic_pose.py)

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
import matplotlib
import matplotlib.pyplot as plt

from .loss import SemanticLossComputer
from .transformer import QK_Norm_TransformerBlock, _init_weights_layerwise
from .transformer import init_weights as _init_weights

from utils.data_utils import SplitData
from utils.pe_utils import get_1d_sincos_pos_emb_from_grid, get_2d_sincos_pos_embed
from utils.pose_utils import rot6d2mat, quat2mat
from utils import camera_utils

DEFAULT_LABELS = ['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other']
NUM_DEFAULT_LABELS = len(DEFAULT_LABELS) + 1  # +1 for background/unknown
if hasattr(matplotlib, "colormaps"):
    DEFAULT_PALETTE = matplotlib.colormaps.get_cmap('tab10').resampled(NUM_DEFAULT_LABELS)
else:
    DEFAULT_PALETTE = plt.cm.get_cmap('tab10', NUM_DEFAULT_LABELS)
DEFAULT_COLORS_LIST = [DEFAULT_PALETTE(i)[:3] for i in range(NUM_DEFAULT_LABELS)]
DEFAULT_COLORS = torch.tensor(DEFAULT_COLORS_LIST, dtype=torch.float32)


class RayZer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.split_data = SplitData(config)

        # Get semantic feature dimension from config (default 512 for LSeg)
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
                * (config.model.image_tokenizer.patch_size**2),
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.image_tokenizer.apply(_init_weights)

        # image positional embedding embedder
        self.use_pe_embedding_layer = config.model.get('input_with_pe', True)
        self.pe_embedder = (
            nn.Sequential(
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
                nn.SiLU(),
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
            )
            if self.use_pe_embedding_layer
            else nn.Identity()
        )
        self.pe_embedder.apply(_init_weights)

        # latent scene representation
        self.scene_code = nn.Parameter(
            torch.randn(
                config.model.scene_latent.length,
                config.model.transformer.d,
            )
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

        # pose pe temporal embedder
        self.temporal_pe_embedder = (
            nn.Sequential(
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
                nn.SiLU(),
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
            )
            if self.use_pe_embedding_layer
            else nn.Identity()
        )
        self.temporal_pe_embedder.apply(_init_weights)

        # qk norm settings
        use_qk_norm = config.model.transformer.get("use_qk_norm", False)

        # transformer encoder and init
        self.transformer_encoder = [
                QK_Norm_TransformerBlock(
                    config.model.transformer.d, config.model.transformer.d_head, use_qk_norm=use_qk_norm
                )
                for _ in range(config.model.transformer.encoder_n_layer)
            ]
        if config.model.transformer.get("special_init", False):
            if config.model.transformer.get('depth_init', False):
                for idx in range(len(self.transformer_encoder)):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    self.transformer_encoder[idx].apply(lambda module: _init_weights_layerwise(module, weight_init_std))
            else:
                for idx in range(len(self.transformer_encoder)):
                    weight_init_std = 0.02 / (2 * config.model.transformer.encoder_n_layer) ** 0.5
                    self.transformer_encoder[idx].apply(lambda module: _init_weights_layerwise(module, weight_init_std))
            self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        else:
            self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
            self.transformer_encoder.apply(_init_weights)

        # transformer encoder2 and init
        self.transformer_encoder_geom = [
                QK_Norm_TransformerBlock(
                    config.model.transformer.d, config.model.transformer.d_head, use_qk_norm=use_qk_norm
                )
                for _ in range(config.model.transformer.encoder_geom_n_layer)
            ]
        if config.model.transformer.get("special_init", False):
            if config.model.transformer.get('depth_init', False):
                for idx in range(len(self.transformer_encoder_geom)):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    self.transformer_encoder_geom[idx].apply(lambda module: _init_weights_layerwise(module, weight_init_std))
            else:
                for idx in range(len(self.transformer_encoder_geom)):
                    weight_init_std = 0.02 / (2 * config.model.transformer.encoder_geom_n_layer) ** 0.5
                    self.transformer_encoder_geom[idx].apply(lambda module: _init_weights_layerwise(module, weight_init_std))
            self.transformer_encoder_geom = nn.ModuleList(self.transformer_encoder_geom)
        else:
            self.transformer_encoder_geom = nn.ModuleList(self.transformer_encoder_geom)
            self.transformer_encoder_geom.apply(_init_weights)

        # ln before decoder
        self.decoder_ln = nn.LayerNorm(config.model.transformer.d, bias=False)

        # transformer decoder and init
        self.transformer_decoder = [
                QK_Norm_TransformerBlock(
                    config.model.transformer.d, config.model.transformer.d_head, use_qk_norm=use_qk_norm
                )
                for _ in range(config.model.transformer.decoder_n_layer)
            ]
        if config.model.transformer.get("special_init", False):
            if config.model.transformer.depth_init:
                for idx in range(len(self.transformer_decoder)):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    self.transformer_decoder[idx].apply(lambda module: _init_weights_layerwise(module, weight_init_std))
            else:
                for idx in range(len(self.transformer_decoder)):
                    weight_init_std = 0.02 / (2 * config.model.transformer.decoder_n_layer) ** 0.5
                    self.transformer_decoder[idx].apply(lambda module: _init_weights_layerwise(module, weight_init_std))
            self.transformer_decoder = nn.ModuleList(self.transformer_decoder)
        else:
            self.transformer_decoder = nn.ModuleList(self.transformer_decoder)
            self.transformer_decoder.apply(_init_weights)

        # pose predictor
        self.pose_predictor = PoseEstimator(config)
        
        # target pose tokenizer
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
                * (config.model.target_image.patch_size**2),
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
                * (config.model.target_image.patch_size**2),
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.target_pose_tokenizer2.apply(_init_weights)

        # fuse mlp
        self.mlp_fuse = nn.Sequential(
            nn.LayerNorm(config.model.transformer.d*2, bias=False),
            nn.Linear(
                config.model.transformer.d*2,
                config.model.transformer.d,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                config.model.transformer.d,
                config.model.transformer.d,
                bias=True,
            ),
        )
        self.mlp_fuse.apply(_init_weights)

        # output regresser
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(config.model.transformer.d, bias=False),
            nn.Linear(
                config.model.transformer.d,
                (config.model.target_image.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid()
        )
        self.image_token_decoder.apply(_init_weights)

        # ============ LSeg Semantic Segmentation Components ============
        self._init_semantic_modules(config)

        # loss (SemanticLossComputer supports both RGB and semantic feature loss)
        self.loss_computer = SemanticLossComputer(config)

        # config backup
        self.config_bk = copy.deepcopy(config)
        self.render_interpolate = config.training.get("render_interpolate", False)

        # training settings
        if config.inference or config.get("evaluation", False):
            if config.training.get('random_split', False):
                self.random_index = True
            else:
                self.random_index = False
        else:
            self.random_index = config.training.get('random_split', False)
        print('Use random index:', self.random_index)

    # ------------------------------------------------------------------ #
    #                   Semantic Module Initialization                     #
    # ------------------------------------------------------------------ #
    def _init_semantic_modules(self, config):
        """Initialize LSeg model and semantic tokenizer / decoder modules."""
        d_model = config.model.transformer.d
        patch_size = config.model.target_image.patch_size
        patch_area = patch_size ** 2

        # --- LSeg feature extractor (frozen) ---
        self._init_lseg()

        if not self.use_lseg:
            return

        # --- Semantic input tokenizer ---
        h, w = config.model.target_image.height, config.model.target_image.width
        sem_pool_k = (h // 2) // (h // patch_size)   # = patch_size // 2
        self.semantic_pool = nn.AvgPool2d(kernel_size=8, stride=8)

        # Project (semantic_feat_dim + 6(plucker)) → d_model
        self.semantic_proj = nn.Linear(self.semantic_feat_dim + 6, d_model)
        self.semantic_proj.apply(_init_weights)

        # --- Target pose tokenizer for semantic output (separate from RGB) ---
        self.target_pose_tokenizer_semantic = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=config.model.target_image.patch_size,
                pw=config.model.target_image.patch_size,
            ),
            nn.Linear(
                config.model.target_image.in_channels
                * (config.model.target_image.patch_size**2),
                d_model,
                bias=False,
            ),
        )
        self.target_pose_tokenizer_semantic.apply(_init_weights)

        # --- Semantic token decoder ---
        # Decode transformer output tokens → intermediate features of d_semantic_hidden per pixel
        self.d_semantic_hidden = 64  # follows LSM design
        output_dim_hidden = patch_area * self.d_semantic_hidden   # e.g. 64 * 64 = 4096
        self.semantic_token_decoder = nn.Sequential(
            nn.LayerNorm(d_model, bias=False),
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 2, output_dim_hidden, bias=False),
        )
        self.semantic_token_decoder.apply(_init_weights)

        # --- Semantic feature expansion ---
        # (d_semantic_hidden, h, w) → (semantic_feat_dim=512, h//2, w//2)
        self.semantic_feature_expansion = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(self.d_semantic_hidden, self.semantic_feat_dim, kernel_size=1, stride=1),
        )
        self.semantic_feature_expansion.apply(_init_weights)

    def _init_lseg(self):
        """Initialize LSeg model for semantic feature extraction."""
        lseg_config = self.config.get("model", {}).get("lseg", {})
        lseg_ckpt_path = lseg_config.get("checkpoint_path", None)

        if lseg_ckpt_path is None:
            print("Warning: LSeg checkpoint_path not provided. Semantic features disabled.")
            self.use_lseg = False
            self.lseg_model = None
            return

        from .lseg import LSegFeatureExtractor
        self.lseg_model = LSegFeatureExtractor.from_pretrained(
            lseg_ckpt_path,
            half_res=True  # output (b, 512, h//2, w//2)
        )
        self.lseg_model.eval()
        for param in self.lseg_model.parameters():
            param.requires_grad = False
        self.use_lseg = True
        print(f"LSeg model loaded from {lseg_ckpt_path}")

    # ------------------------------------------------------------------ #
    #                          LSeg helpers                                #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract_lseg_features(self, images, plucker_half=None):
        """
        Extract LSeg features from images, optionally with Plücker pose conditioning.

        Args:
            images: [b, v, 3, h, w], value range [0, 1]
            plucker_half: [b*v, 6, h//2, w//2], Plücker rays at half resolution (optional)

        Returns:
            lseg_token_feature: [b, v*n_sem, feat_dim+6] or [b, v*n_sem, feat_dim]
            lseg_res_feature:   [b*v, 512, h//2, w//2]
        """
        if self.lseg_model is None:
            return None, None

        b, v, c, h, w = images.shape
        images_flat = images.reshape(b * v, c, h, w)

        lseg_features = self.lseg_model.extract_features(images_flat)  # [b*v, 512, h//2, w//2]

        if plucker_half is not None:
            lseg_with_pose = torch.cat([lseg_features, plucker_half], dim=1)  # [b*v, 518, h//2, w//2]
        else:
            lseg_with_pose = lseg_features

        # Pool to align with image patch grid
        features = self.semantic_pool(lseg_with_pose)  # [b*v, 518, h_pool, w_pool]
        lseg_token_feature = rearrange(features, '(b v) c h w -> b (v h w) c', b=b, v=v)

        return lseg_token_feature, lseg_features  # lseg_features as residual

    # ------------------------------------------------------------------ #
    #                          Train / Eval                                #
    # ------------------------------------------------------------------ #
    def train(self, mode=True):
        super().train(mode)
        self.loss_computer.eval()
        if self.use_lseg and self.lseg_model is not None:
            self.lseg_model.eval()

    def get_overview(self):
        count_train_params = lambda model: sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        overview = edict(
            image_tokenizer=count_train_params(self.image_tokenizer),
            pe_embedder=count_train_params(self.pe_embedder),
            temporal_pe_embedder=count_train_params(self.temporal_pe_embedder),
            scene_code=self.scene_code.data.numel(),
            cam_code=self.cam_code.data.numel(),
            transformer_encoder=count_train_params(self.transformer_encoder),
            transformer_encoder_geom=count_train_params(self.transformer_encoder_geom),
            transformer_decoder=count_train_params(self.transformer_decoder),
            mlp_fuse=count_train_params(self.mlp_fuse),
            target_pose_tokenizer=count_train_params(self.target_pose_tokenizer),
            target_pose_tokenizer2=count_train_params(self.target_pose_tokenizer2),
            image_token_decoder=count_train_params(self.image_token_decoder),
            pose_predictor=count_train_params(self.pose_predictor),
        )
        if self.use_lseg:
            overview.semantic_proj = count_train_params(self.semantic_proj)
            overview.target_pose_tokenizer_semantic = count_train_params(self.target_pose_tokenizer_semantic)
            overview.semantic_token_decoder = count_train_params(self.semantic_token_decoder)
            overview.semantic_feature_expansion = count_train_params(self.semantic_feature_expansion)
        return overview

    # ------------------------------------------------------------------ #
    #                          Forward                                     #
    # ------------------------------------------------------------------ #
    def forward(self, data, create_visual=False, render_video=False, iter=0):

        '''Split all images into two sets, use one set to get scene representation, use the other to render & train'''
        input, target, input_idx, target_idx = self.split_data(data, random_index=self.random_index)
        image = input.image * 2.0 - 1.0                                           # [b, v, c, h, w], range (0,1) to (-1,1)
        b, v_input, c, h, w = image.shape
        image_all = data['image'] * 2.0 - 1.0                                     # [b, v_all, c, h, w], range (0,1) to (-1,1)
        v_all = image_all.shape[1]
        v_target = v_all - v_input
        device = image.device
        input_idx, target_idx = input_idx.to(device), target_idx.to(device)
        batch_idx = torch.arange(b).unsqueeze(1).to(device)

        '''se3 pose prediction for all views'''
        # tokenize images, add spatial-temporal p.e.
        img_tokens = self.image_tokenizer(image_all)                             # [b * v, n, d]
        _, n, d = img_tokens.shape
        if self.use_pe_embedding_layer:
            img_tokens = self.add_sptial_temporal_pe(img_tokens, b, v_all, h, w)
        img_tokens = rearrange(img_tokens, '(b v) n d -> b (v n) d', b=b, v=v_all)   # [b, v * n, d]

        # get camera tokens, add temporal p.e.
        cam_tokens = self.get_camera_tokens(b, v_all)                            # [b, v_all * n_cam, d]
        n_cam = cam_tokens.shape[1] // v_all
        assert n_cam == 1
        cam_tokens = rearrange(cam_tokens, 'b (v n) d -> b v n d', v=v_all)      # [b, v_all, n_cam, d]
        cam_tokens = rearrange(cam_tokens, 'b v n d -> b (v n) d')               # [b, v_all * n_cam, d]

        # pose estimation for all views
        all_tokens = torch.cat([cam_tokens, img_tokens], dim=1)
        all_tokens = self.run_encoder(all_tokens)
        cam_tokens, _ = all_tokens.split([v_all * n_cam, v_all * n], dim=1)

        # get se3 poses and intrinsics
        cam_tokens = rearrange(cam_tokens, 'b (v n) d -> (b v) n d', b=b, v=v_all, n=n_cam)[:, 0]    # [b * v_all, d]
        cam_info = self.pose_predictor(cam_tokens, v_all)                        # [b * v_all, num_pose_element+3+4]
        c2w, fxfycxcy = get_cam_se3(cam_info)                                    # [b * v_all,4,4], [b * v_all,4]
        normalized = True

        # get plucker ray and embeddings
        plucker_rays = cam_info_to_plucker(c2w, fxfycxcy, self.config.model.target_image, normalized=normalized)
        plucker_rays = rearrange(plucker_rays, '(b v) c h w -> b v c h w', b=b, v=v_all)
        plucker_emb_input = self.target_pose_tokenizer(plucker_rays[batch_idx, input_idx])         # [b * v_input, n, d]
        plucker_emb_target = self.target_pose_tokenizer2(plucker_rays[batch_idx, target_idx])      # [b * v_target, n, d]
        plucker_emb_input = rearrange(plucker_emb_input, '(b v) n d -> b (v n) d', v=v_input)      # [b, v_input * n, d]

        '''predict scene representation using (posed) input views'''
        # get posed image representation
        img_tokens_input = rearrange(img_tokens, 'b (v n) d -> b v n d', v=v_all)[batch_idx, input_idx]       # [b, v_input, n, d]
        img_tokens_input = rearrange(img_tokens_input, 'b v n d -> b (v n) d')
        img_tokens_input = torch.cat([img_tokens_input, plucker_emb_input], dim=-1)      # [b, v_input * n, 2d]
        img_tokens_input = self.mlp_fuse(img_tokens_input)                               # [b, v_input * n, d]

        # --- Extract LSeg semantic tokens from input views ---
        input_sem_tokens = None
        if self.use_lseg:
            # Plucker rays at half resolution for input views
            plucker_input_views = plucker_rays[batch_idx, input_idx]  # [b, v_input, 6, h, w]
            plucker_input_flat = rearrange(plucker_input_views, 'b v c h w -> (b v) c h w')
            plucker_input_half = F.interpolate(
                plucker_input_flat, scale_factor=0.5, mode='bilinear', align_corners=False
            )  # [b*v_input, 6, h//2, w//2]

            # Extract LSeg features
            lseg_token_feature, _ = self.extract_lseg_features(
                input.image, plucker_half=plucker_input_half
            )  # [b, v_input*n_sem, 518]

            # Project 518 → d_model
            input_sem_tokens = self.semantic_proj(lseg_token_feature)  # [b, v_input*n_sem, d]

        # replicate scene tokens
        scene_tokens = self.scene_code.expand(b, -1, -1)                         # [b, n_scene, d]
        n_scene = scene_tokens.shape[1]

        # concat scene tokens, image tokens, and semantic tokens
        if input_sem_tokens is not None:
            n_sem = input_sem_tokens.shape[1]
            all_tokens = torch.cat([scene_tokens, img_tokens_input, input_sem_tokens], dim=1)
        else:
            n_sem = 0
            all_tokens = torch.cat([scene_tokens, img_tokens_input], dim=1)

        # encoder layers, update scene representation
        all_tokens = self.run_encoder_geom(all_tokens)
        scene_tokens, _ = all_tokens.split([n_scene, v_input * n + n_sem], dim=1)

        '''render with scene representation from input views and pose of target views'''
        # --- Build target tokens (RGB + semantic) ---
        if self.use_lseg:
            plucker_emb_target_sem = self.target_pose_tokenizer_semantic(
                plucker_rays[batch_idx, target_idx]
            )  # [b * v_target, n, d]
        else:
            plucker_emb_target_sem = None

        render_results = self.render_images(
            scene_tokens, plucker_emb_target, plucker_emb_target_sem
        )

        if create_visual and render_video:
            with torch.no_grad():
                c2w_target = rearrange(c2w, '(b v) c d -> b v c d', v=v_all)[batch_idx, target_idx]
                fxfycxcy_target = rearrange(fxfycxcy, '(b v) c -> b v c', v=v_all)[batch_idx, target_idx]
                c2w_target = rearrange(c2w_target, 'b v c d -> (b v) c d')
                fxfycxcy_target = rearrange(fxfycxcy_target, 'b v c -> (b v) c')
                vis_only_results = self.render_images_video(scene_tokens, c2w_target, fxfycxcy_target, normalized=normalized)

        # compute loss (RGB + semantic feature)
        rendered_sem_for_loss = None
        gt_lseg_features = None
        if self.use_lseg and render_results.get('rendered_sem_features') is not None:
            rendered_sem_for_loss = render_results.rendered_sem_features  # [b*v_target, 512, h//2, w//2]
            # Extract GT LSeg features from target images
            target_images_flat = target.image.reshape(-1, 3, h, w)
            gt_lseg_features = self.lseg_model.extract_features(target_images_flat)  # [b*v_target, 512, h//2, w//2]

        loss_metrics = self.loss_computer(
            render_results.rendered_images,
            target.image,
            rendered_sem_features=rendered_sem_for_loss,
            target_sem_features=gt_lseg_features,
        )

        # return results
        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=render_results.rendered_images,
            render_semantic=render_results.get('rendered_sem_features'),
            c2w=rearrange(c2w, '(b v) c d -> b v c d', b=b, v=v_all),
            fxfycxcy=fxfycxcy,
            input_idx=input_idx,
            target_idx=target_idx
        )

        if create_visual and render_video:
            result.video_rendering = vis_only_results.rendered_images_video.detach()

        return result

    # ------------------------------------------------------------------ #
    #                      Render helpers                                   #
    # ------------------------------------------------------------------ #
    def render_images(self, scene_tokens, target_tokens_rgb, target_tokens_sem=None):
        """
        Render target views.

        Args:
            scene_tokens:      [b, n_scene, d]
            target_tokens_rgb: [b*v, n_target, d]  — Plücker embeddings for RGB
            target_tokens_sem: [b*v, n_target, d]  — Plücker embeddings for semantic (optional)
        """
        b, _, d = scene_tokens.shape
        bv = target_tokens_rgb.shape[0]
        v = bv // b

        # repeat scene tokens
        scene_tokens = scene_tokens.unsqueeze(1).repeat(1, v, 1, 1)
        scene_tokens = rearrange(scene_tokens, 'b v n d -> (b v) n d')

        n_target = target_tokens_rgb.shape[1]
        n_scene = scene_tokens.shape[1]

        if target_tokens_sem is not None:
            n_sem_target = target_tokens_sem.shape[1]
            all_tokens = torch.cat([target_tokens_rgb, target_tokens_sem, scene_tokens], dim=1)
        else:
            n_sem_target = 0
            all_tokens = torch.cat([target_tokens_rgb, scene_tokens], dim=1)

        return self.render(all_tokens, n_target, n_sem_target, n_scene, v)

    def render(self, all_tokens, n_target, n_sem_target, n_scene, v):
        '''
        Run decoder layers and output heads.

        Args:
            all_tokens: [b*v, n_target + n_sem_target + n_scene, d]
        '''
        all_tokens = self.decoder_ln(all_tokens)
        all_tokens = self.run_decoder(all_tokens)

        # split tokens
        if n_sem_target > 0:
            target_rgb_tokens, target_sem_tokens, _ = all_tokens.split(
                [n_target, n_sem_target, n_scene], dim=1
            )
        else:
            target_rgb_tokens, _ = all_tokens.split([n_target, n_scene], dim=1)
            target_sem_tokens = None

        patch_size = self.config.model.target_image.patch_size

        # --- RGB head ---
        rendered_images_all = self.image_token_decoder(target_rgb_tokens)
        rendered_images_all = rearrange(
            rendered_images_all, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v,
            h=self.target_latent_h, w=self.target_latent_w,
            p1=patch_size, p2=patch_size, c=3
        )

        render_results = edict(rendered_images=rendered_images_all)

        # --- Semantic head ---
        if self.use_lseg and target_sem_tokens is not None:
            decoded_sem_hidden = self.semantic_token_decoder(target_sem_tokens)   # [b*v, n_target, p*p*64]
            decoded_sem_hidden = rearrange(
                decoded_sem_hidden, "(b v) (h w) (p1 p2 c) -> (b v) c (h p1) (w p2)",
                v=v,
                h=self.target_latent_h, w=self.target_latent_w,
                p1=patch_size, p2=patch_size, c=self.d_semantic_hidden
            )  # [b*v, 64, h, w]

            # Expand to 512-dim and downsample to half resolution (align with GT LSeg)
            rendered_sem_features = self.semantic_feature_expansion(decoded_sem_hidden)  # [b*v, 512, h//2, w//2]
            render_results.rendered_sem_features = rendered_sem_features

        return render_results

    # ------------------------------------------------------------------ #
    #                   Semantic decoding (inference)                       #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode_semantic_to_segmentation(self, semantic_features, labels=None):
        """
        Decode semantic features into segmentation maps using LSeg text-image similarity.

        Args:
            semantic_features: [b, 512, h, w]
            labels: list of semantic label strings

        Returns:
            segmentation:     [b, h, w] label indices (1-indexed, 0=background)
            segmentation_vis: [b, 3, h, w] colour visualisation
        """
        if not self.use_lseg or self.lseg_model is None:
            return None, None

        if labels is None:
            labels = DEFAULT_LABELS

        logits = self.lseg_model.decode_feature(semantic_features, labelset=labels)
        segmentation = torch.argmax(logits, dim=1) + 1  # +1: 0 reserved for bg

        num_labels = len(labels) + 1
        if num_labels == NUM_DEFAULT_LABELS:
            colors = DEFAULT_COLORS
        else:
            if hasattr(matplotlib, "colormaps"):
                palette = matplotlib.colormaps.get_cmap('tab10').resampled(num_labels)
            else:
                palette = plt.cm.get_cmap('tab10', num_labels)
            colors_list = [palette(i)[:3] for i in range(num_labels)]
            colors = torch.tensor(colors_list, dtype=torch.float32)

        segmentation_vis = colors[segmentation.cpu()]  # [b, h, w, 3]
        segmentation_vis = rearrange(segmentation_vis, 'b h w c -> b c h w')

        return segmentation, segmentation_vis

    # ------------------------------------------------------------------ #
    #                     Positional encoding helpers                       #
    # ------------------------------------------------------------------ #
    def get_camera_tokens(self, b, v):
        n, d = self.cam_code.shape[-2:]
        cam_tokens = rearrange(self.cam_code, 'n d -> 1 1 n d')
        cam_tokens = repeat(cam_tokens, '1 1 n d -> 1 v n d', v=v)
        cam_tokens = rearrange(cam_tokens, '1 v n d -> 1 (v n) d')

        img_indices = torch.arange(v).repeat_interleave(n)
        img_indices = img_indices.to(cam_tokens.device)
        temporal_pe = get_1d_sincos_pos_emb_from_grid(
            embed_dim=d, pos=img_indices, device=cam_tokens.device
        ).to(cam_tokens.dtype)
        temporal_pe = temporal_pe.reshape(1, v * n, d)
        temporal_pe = self.temporal_pe_embedder(temporal_pe)

        return (cam_tokens + temporal_pe).repeat(b, 1, 1)

    def add_sptial_temporal_pe(self, img_tokens, b, v, h_origin, w_origin):
        patch_size = self.config.model.image_tokenizer.patch_size
        num_h_tokens = h_origin // patch_size
        num_w_tokens = w_origin // patch_size
        assert (num_h_tokens * num_w_tokens) == img_tokens.shape[1]
        bv, n, d = img_tokens.shape

        img_indices = torch.arange(v).repeat_interleave(n)
        img_indices = img_indices.unsqueeze(0).repeat(b, 1).reshape(-1)
        img_indices = img_indices.to(img_tokens.device)
        temporal_pe = get_1d_sincos_pos_emb_from_grid(
            embed_dim=d // 2, pos=img_indices, device=img_tokens.device
        ).to(img_tokens.dtype)
        temporal_pe = temporal_pe.reshape(b, v, n, d // 2)

        spatial_pe = get_2d_sincos_pos_embed(
            embed_dim=d // 2, grid_size=(num_h_tokens, num_w_tokens), device=img_tokens.device
        ).to(img_tokens.dtype)
        spatial_pe = spatial_pe.reshape(1, 1, n, d // 2).repeat(b, v, 1, 1)

        pe = self.pe_embedder(
            torch.cat([spatial_pe, temporal_pe], dim=-1).reshape(bv, n, d)
        )
        return img_tokens + pe

    # ------------------------------------------------------------------ #
    #                     Transformer layer runners                        #
    # ------------------------------------------------------------------ #
    def run_layers_encoder(self, start, end):
        def custom_forward(tokens):
            for i in range(start, min(end, len(self.transformer_encoder))):
                tokens = self.transformer_encoder[i](tokens)
            return tokens
        return custom_forward

    def run_layers_encoder_geom(self, start, end):
        def custom_forward(tokens):
            for i in range(start, min(end, len(self.transformer_encoder_geom))):
                tokens = self.transformer_encoder_geom[i](tokens)
            return tokens
        return custom_forward

    def run_layers_decoder(self, start, end):
        def custom_forward(tokens):
            for i in range(start, min(end, len(self.transformer_decoder))):
                tokens = self.transformer_decoder[i](tokens)
            return tokens
        return custom_forward

    def run_encoder(self, all_tokens_encoder):
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_encoder), checkpoint_every):
            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_encoder(i, i + 1), all_tokens_encoder, use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_encoder(i + 1, i + checkpoint_every)(all_tokens_encoder)
        return all_tokens_encoder

    def run_encoder_geom(self, all_tokens_encoder):
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_encoder_geom), checkpoint_every):
            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_encoder_geom(i, i + 1), all_tokens_encoder, use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_encoder_geom(i + 1, i + checkpoint_every)(all_tokens_encoder)
        return all_tokens_encoder

    def run_decoder(self, all_tokens_encoder):
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_decoder), checkpoint_every):
            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_decoder(i, i + 1), all_tokens_encoder, use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_decoder(i + 1, i + checkpoint_every)(all_tokens_encoder)
        return all_tokens_encoder

    # ------------------------------------------------------------------ #
    #                      Video rendering                                 #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def render_images_video(self, scene_tokens_all, c2w_all, fxfycxcy_all, normalized=False):
        '''
        scene_tokens_all: [b, n_scene, d]
        c2w_all: [b*v, 4, 4]
        fxfycxcy_all: [b*v, 4]
        '''
        scene_tokens_all = scene_tokens_all.detach()
        c2w_all = c2w_all.detach()
        fxfycxcy_all = fxfycxcy_all.detach()

        b, _, d = scene_tokens_all.shape
        bv = c2w_all.shape[0]
        v = bv // b
        c2w_all = rearrange(c2w_all, '(b v) x y -> b v x y', v=v)
        fxfycxcy_all = rearrange(fxfycxcy_all, '(b v) x -> b v x', v=v)
        device = scene_tokens_all.device

        all_renderings = []
        all_sem_features = [] if self.use_lseg else None
        num_frames = self.config.inference.render_video_config.num_frames
        traj_type = self.config.inference.render_video_config.traj_type
        order_poses = False

        for i in range(b):
            scene_tokens = scene_tokens_all[i]
            c2ws = c2w_all[i]
            fxfycxcy = fxfycxcy_all[i]

            if traj_type == "interpolate":
                Ks = torch.zeros((c2ws.shape[0], 3, 3), device=device)
                Ks[:, 0, 0] = fxfycxcy[:, 0]
                Ks[:, 1, 1] = fxfycxcy[:, 1]
                Ks[:, 0, 2] = fxfycxcy[:, 2]
                Ks[:, 1, 2] = fxfycxcy[:, 3]
                c2ws, Ks = camera_utils.get_interpolated_poses_many(
                    c2ws[:, :3, :4], Ks, num_frames, order_poses=order_poses
                )
                frame_c2ws = torch.cat([
                    c2ws.to(device),
                    torch.tensor([[[0, 0, 0, 1]]], device=device).repeat(c2ws.shape[0], 1, 1)
                ], dim=1)
                frame_fxfycxcy = torch.zeros((c2ws.shape[0], 4), device=device)
                frame_fxfycxcy[:, 0] = Ks[:, 0, 0]
                frame_fxfycxcy[:, 1] = Ks[:, 1, 1]
                frame_fxfycxcy[:, 2] = Ks[:, 0, 2]
                frame_fxfycxcy[:, 3] = Ks[:, 1, 2]
            elif traj_type == "same":
                frame_c2ws = c2ws.clone()
                frame_fxfycxcy = fxfycxcy.clone()
            else:
                raise NotImplementedError

            plucker_rays = cam_info_to_plucker(
                frame_c2ws, frame_fxfycxcy, self.config.model.target_image, normalized=normalized
            )  # [v', 6, h, w]
            plucker_embeddings_rgb = self.target_pose_tokenizer2(plucker_rays.unsqueeze(0))  # [v', n_target, d]

            if self.use_lseg:
                plucker_embeddings_sem = self.target_pose_tokenizer_semantic(plucker_rays.unsqueeze(0))
            else:
                plucker_embeddings_sem = None

            v_render = plucker_embeddings_rgb.shape[0]
            scene_rep = scene_tokens.unsqueeze(0).repeat(v_render, 1, 1)
            n_target = plucker_embeddings_rgb.shape[1]
            n_scene_cur = scene_rep.shape[1]

            if plucker_embeddings_sem is not None:
                n_sem_target = plucker_embeddings_sem.shape[1]
                all_tokens = torch.cat([plucker_embeddings_rgb, plucker_embeddings_sem, scene_rep], dim=1)
            else:
                n_sem_target = 0
                all_tokens = torch.cat([plucker_embeddings_rgb, scene_rep], dim=1)

            render_outputs = self.render(all_tokens, n_target, n_sem_target, n_scene_cur, v_render)
            rendered_images = render_outputs.rendered_images.squeeze(0)  # [v', c, h, w]
            all_renderings.append(rendered_images)

            if self.use_lseg and render_outputs.get('rendered_sem_features') is not None:
                all_sem_features.append(render_outputs.rendered_sem_features)  # [v', 512, h//2, w//2]

        all_renderings = torch.stack(all_renderings)  # [b, v', c, h, w]

        render_results = edict(rendered_images_video=all_renderings)
        if self.use_lseg and all_sem_features:
            render_results.video_semantic = torch.stack(all_sem_features)
        return render_results

    @torch.no_grad()
    def render_video_with_segmentation(self, scene_tokens_all, c2w_all, fxfycxcy_all,
                                        normalized=False, labels=None):
        """
        Render video with both RGB and semantic segmentation visualisation.

        Returns edict with:
            video_rendering:          [b, v', 3, h, w]
            video_semantic:           [b, v', 512, h//2, w//2]
            video_segmentation:       [b, v', h, w]
            video_segmentation_vis:   [b, v', 3, h, w]
        """
        if labels is None:
            labels = DEFAULT_LABELS

        result = self.render_images_video(scene_tokens_all, c2w_all, fxfycxcy_all, normalized=normalized)

        if self.use_lseg and result.get('video_semantic') is not None:
            video_sem = result.video_semantic  # [b, v', 512, h//2, w//2]
            b, vp = video_sem.shape[:2]
            sem_flat = rearrange(video_sem, 'b v c h w -> (b v) c h w')

            segmentation, segmentation_vis = self.decode_semantic_to_segmentation(sem_flat, labels=labels)

            if segmentation is not None:
                result.video_segmentation = rearrange(segmentation, '(b v) h w -> b v h w', b=b, v=vp)
            if segmentation_vis is not None:
                result.video_segmentation_vis = rearrange(segmentation_vis, '(b v) c h w -> b v c h w', b=b, v=vp)

        return result

    # ------------------------------------------------------------------ #
    #                       Checkpoint loading                             #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [f for f in os.listdir(load_path) if f.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, n) for n in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except Exception:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None

        self.load_state_dict(checkpoint["model"], strict=False)
        return 0


# ====================================================================== #
#                         Utility functions                                #
# ====================================================================== #
def get_cam_se3(cam_info):
    '''cam_info: [b, num_pose_element+3+4]'''
    b, n = cam_info.shape

    if n == 13:
        rot_6d = cam_info[:, :6]
        R = rot6d2mat(rot_6d)
        t = cam_info[:, 6:9].unsqueeze(-1)
        fxfycxcy = cam_info[:, 9:]
    elif n == 11:
        rot_quat = cam_info[:, :4]
        R = quat2mat(rot_quat)
        t = cam_info[:, 4:7].unsqueeze(-1)
        fxfycxcy = cam_info[:, 7:]
    else:
        raise NotImplementedError

    Rt = torch.cat([R, t], dim=2)
    c2w = torch.cat([
        Rt,
        torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device).view(1, 1, 4).repeat(b, 1, 1)
    ], dim=1)
    return c2w, fxfycxcy


def cam_info_to_plucker(c2w, fxfycxcy, target_imgs_info, normalized=True):
    '''c2w: [b,4,4], fxfycxcy: [b,4]'''
    b = c2w.shape[0]
    device = c2w.device
    h, w = target_imgs_info.height, target_imgs_info.width

    fxfycxcy = fxfycxcy.clone()
    if normalized:
        fxfycxcy[:, 0] *= w
        fxfycxcy[:, 1] *= h
        fxfycxcy[:, 2] *= w
        fxfycxcy[:, 3] *= h

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(c2w), x.to(c2w)
    x = x[None, :, :].expand(b, -1, -1).reshape(b, -1)
    y = y[None, :, :].expand(b, -1, -1).reshape(b, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)

    ray_o = ray_o.reshape(b, h, w, 3).permute(0, 3, 1, 2)
    ray_d = ray_d.reshape(b, h, w, 3).permute(0, 3, 1, 2)

    plucker = torch.cat([torch.cross(ray_o, ray_d, dim=1), ray_d], dim=1)
    return plucker  # [b, 6, h, w]


# ====================================================================== #
#                         Pose Estimator                                   #
# ====================================================================== #
class PoseEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pose_rep = self.config.model.pose_latent.get('representation', '6d')
        print('Pose representation:', self.pose_rep)
        if self.pose_rep == '6d':
            self.num_pose_element = 6
        elif self.pose_rep == 'quat':
            self.num_pose_element = 4
        else:
            raise NotImplementedError

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.rel_head = nn.Sequential(
            nn.Linear(config.model.transformer.d * 2, config.model.transformer.d, bias=True),
            nn.SiLU(),
            nn.Linear(config.model.transformer.d, self.num_pose_element + 3, bias=True),
        )
        self.rel_head.apply(init_weights)

        self.canonical_k_head = nn.Sequential(
            nn.Linear(config.model.transformer.d, config.model.transformer.d, bias=True),
            nn.SiLU(),
            nn.Linear(config.model.transformer.d, 1, bias=True),
        )
        self.canonical_k_head.apply(init_weights)
        self.f_bias = 1.25

    def forward(self, x, v):
        '''x: [b*v, d]'''
        canonical = self.config.model.pose_latent.get('canonical', 'first')
        x = rearrange(x, '(b v) d -> b v d', v=v)
        b = x.shape[0]

        if canonical == 'first':
            x_canonical = x[:, 0:1]
            x_rel = x[:, 1:]
        elif canonical == 'middle':
            cano_idx = v // 2
            rel_indices = torch.cat([torch.arange(cano_idx), torch.arange(cano_idx + 1, v)])
            x_canonical = x[:, cano_idx:cano_idx + 1]
            x_rel = x[:, rel_indices]
        else:
            raise NotImplementedError

        fxfy_canonical = self.canonical_k_head(x_canonical[:, 0]) + self.f_bias
        fxfy_canonical = fxfy_canonical.unsqueeze(1).repeat(1, 1, 2)

        if self.pose_rep == '6d':
            rt_canonical = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(1, 1, 9).to(fxfy_canonical).repeat(b, 1, 1)
        elif self.pose_rep == 'quat':
            rt_canonical = torch.tensor([1, 0, 0, 0, 0, 0, 0]).reshape(1, 1, 7).to(fxfy_canonical).repeat(b, 1, 1)
        info_canonical = torch.cat([rt_canonical, fxfy_canonical], dim=-1)

        feat_rel = torch.cat([x_canonical.repeat(1, v - 1, 1), x_rel], dim=-1)
        info_rel = self.rel_head(feat_rel)
        info_all = info_canonical.repeat(1, v, 1)

        if canonical == 'first':
            info_all[:, 1:, :self.num_pose_element + 3] += info_rel
        elif canonical == 'middle':
            info_all[:, rel_indices, :self.num_pose_element + 3] += info_rel
        else:
            raise NotImplementedError

        cxcy_all = torch.tensor([0.5, 0.5]).reshape(1, 1, 2).repeat(b, v, 1).to(info_all)
        info_all = torch.cat([info_all, cxcy_all], dim=-1)
        return rearrange(info_all, 'b v d -> (b v) d')
