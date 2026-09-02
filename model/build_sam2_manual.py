"""Construct the optional SAM2 refinement model without Hydra."""

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)


def _ensure_sam2_importable():
    """Expose the vendored SAM2 package to its absolute imports."""
    model_dir = os.path.dirname(os.path.abspath(__file__))
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)


def build_sam2_manual(
    config_name: str = "sam2.1_hiera_l",
    ckpt_path: str = None,
    device: str = "cuda",
    mode: str = "eval",
):
    """Build a supported SAM2 hierarchy and optionally load its checkpoint."""
    _ensure_sam2_importable()

    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
    from sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
    from sam2.modeling.sam.transformer import RoPEAttention
    from sam2.modeling.sam2_base import SAM2Base

    if "hiera_l" in config_name:
        hiera_kwargs = dict(
            embed_dim=144,
            num_heads=2,
            stages=[2, 6, 36, 4],
            global_att_blocks=[23, 33, 43],
            window_pos_embed_bkg_spatial_size=[7, 7],
            window_spec=[8, 4, 16, 8],
        )
        backbone_channel_list = [1152, 576, 288, 144]
    elif "hiera_b+" in config_name or "hiera_b" in config_name:
        hiera_kwargs = dict(
            embed_dim=112,
            num_heads=2,
            stages=[2, 3, 16, 3],
            global_att_blocks=[12, 16, 20],
            window_pos_embed_bkg_spatial_size=[14, 14],
            window_spec=[8, 4, 14, 7],
        )
        backbone_channel_list = [896, 448, 224, 112]
    elif "hiera_s" in config_name:
        hiera_kwargs = dict(
            embed_dim=96,
            num_heads=1,
            stages=[1, 2, 11, 2],
            global_att_blocks=[7, 10, 13],
            window_pos_embed_bkg_spatial_size=[14, 14],
            window_spec=[8, 4, 14, 7],
        )
        backbone_channel_list = [768, 384, 192, 96]
    elif "hiera_t" in config_name:
        hiera_kwargs = dict(
            embed_dim=96,
            num_heads=1,
            stages=[1, 2, 7, 2],
            global_att_blocks=[5, 7, 9],
            window_pos_embed_bkg_spatial_size=[14, 14],
            window_spec=[8, 4, 14, 7],
        )
        backbone_channel_list = [768, 384, 192, 96]
    else:
        raise ValueError(
            f"Unsupported config_name: {config_name}. "
            "Supported: hiera_l, hiera_b+, hiera_s, hiera_t"
        )

    # ---- 1. Image Encoder ----
    trunk = Hiera(**hiera_kwargs)

    pos_enc_neck = PositionEmbeddingSine(
        num_pos_feats=256, normalize=True, scale=None, temperature=10000,
    )
    neck = FpnNeck(
        position_encoding=pos_enc_neck,
        d_model=256,
        backbone_channel_list=backbone_channel_list,
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )
    image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1)

    # ---- 2. Memory Attention ----
    self_attention = RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=[64, 64],
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
    )
    cross_attention = RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=[64, 64],
        rope_k_repeat=True,
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
    )
    mem_attn_layer = MemoryAttentionLayer(
        activation="relu",
        cross_attention=cross_attention,
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention=self_attention,
    )
    memory_attention = MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=mem_attn_layer,
        num_layers=4,
    )

    # ---- 3. Memory Encoder ----
    pos_enc_mem = PositionEmbeddingSine(
        num_pos_feats=64, normalize=True, scale=None, temperature=10000,
    )
    mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)
    cx_block = CXBlock(
        dim=256, kernel_size=7, padding=3,
        layer_scale_init_value=1e-6, use_dwconv=True,
    )
    fuser = Fuser(layer=cx_block, num_layers=2)
    memory_encoder = MemoryEncoder(
        out_dim=64,
        position_encoding=pos_enc_mem,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    # ---- 4. SAM2Base ----
    sam2_extra_args = dict(
        dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
    )

    model = SAM2Base(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        num_maskmem=7,
        image_size=1024,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        no_obj_embed_spatial=True,
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_signed_tpos_enc_to_obj_ptrs=True,
        only_obj_ptrs_in_the_past_for_eval=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
        compile_image_encoder=False,
        sam_mask_decoder_extra_args=sam2_extra_args,
    )

    if ckpt_path is not None and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logger.warning(f"SAM2 missing keys: {missing_keys[:10]}...")
        if unexpected_keys:
            logger.warning(f"SAM2 unexpected keys: {unexpected_keys[:10]}...")
        logger.info(f"SAM2 checkpoint loaded from {ckpt_path}")
    elif ckpt_path:
        logger.warning(f"SAM2 checkpoint not found: {ckpt_path}")

    model = model.to(device)
    if mode == "eval":
        model.eval()

    return model
