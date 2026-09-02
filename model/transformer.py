# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from einops import rearrange
import os

USE_FLASH_ATTENTION = os.environ.get('USE_FLASH_ATTENTION', '1') == '1'

if USE_FLASH_ATTENTION:
    try:
        from flash_attn import flash_attn_func
        FLASH_ATTN_AVAILABLE = True
    except ImportError:
        FLASH_ATTN_AVAILABLE = False
        print("Warning: flash_attn not available, will use xformers or scaled_dot_product_attention as fallback")
else:
    FLASH_ATTN_AVAILABLE = False
    print("Flash attention disabled via USE_FLASH_ATTENTION=0, using xformers or scaled_dot_product_attention")

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False



def init_weights(module, std=0.02):
    """Initialize weights for linear and embedding layers.
    
    Args:
        module: Module to initialize
        std: Standard deviation for normal initialization
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)



# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        orig_dtype = x.dtype
        output = self._norm(x.float()).to(orig_dtype)
        return output * self.weight.to(orig_dtype)



class MLP(nn.Module):
    """
    Multi-Layer Perceptron block.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L49-L65
    """
    
    def __init__(
        self,
        dim,
        mlp_ratio=4,
        bias=False,
        dropout=0.0,
        activation=nn.GELU,
        mlp_dim=None,
    ):
        """
        Args:
            dim: Input dimension
            mlp_ratio: Multiplier for hidden dimension
            bias: Whether to use bias in linear layers
            dropout: Dropout probability
            activation: Activation function
            mlp_dim: Optional explicit hidden dimension (overrides mlp_ratio)
        """
        super().__init__()
        hidden_dim = mlp_dim if mlp_dim is not None else int(dim * mlp_ratio)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            activation(),
            nn.Linear(hidden_dim, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)



class QK_Norm_SelfAttention(nn.Module):
    """
    Self-attention with optional Q-K normalization.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        dim,
        head_dim,
        qkv_bias=False,
        fc_bias=True,
        attn_dropout=0.0,
        fc_dropout=0.0,
        use_qk_norm=True,
    ):
        """
        Args:
            dim: Input dimension
            head_dim: Dimension of each attention head
            qkv_bias: Whether to use bias in QKV projection
            fc_bias: Whether to use bias in output projection
            attn_dropout: Dropout probability for attention weights
            fc_dropout: Dropout probability for output projection
            use_qk_norm: Whether to use Q-K normalization
        We use flash attention V2 for efficiency.
        """
        super().__init__()
        assert dim % head_dim == 0, f"Token dimension {dim} should be divisible by head dimension {head_dim}"
        
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.attn_dropout = attn_dropout
        self.use_qk_norm = use_qk_norm

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.fc = nn.Linear(dim, dim, bias=fc_bias)
        self.attn_fc_dropout = nn.Dropout(fc_dropout)
        
        # Optional Q-K normalization
        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x, attn_bias=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attn_bias: Optional attention bias mask
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        q, k, v = (rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim) for t in (q, k, v))
        
        # Apply qk normalization if enabled
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
            
        if torch.is_autocast_enabled():
            amp_dtype = torch.get_autocast_gpu_dtype()
            q = q.to(amp_dtype)
            k = k.to(amp_dtype)
            v = v.to(amp_dtype)

        dropout_p = self.attn_dropout if self.training else 0.0
        
        if FLASH_ATTN_AVAILABLE and q.is_cuda:
            x = flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=None,
                causal=False,
            )
        elif XFORMERS_AVAILABLE and q.is_cuda:
            # Fallback to xformers
            x = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                p=dropout_p,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
        else:
            # Fallback to PyTorch native scaled_dot_product_attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=dropout_p,
                is_causal=False,
            )
            x = x.transpose(1, 2)
        
        x = rearrange(x, "b l nh dh -> b l (nh dh)")
        x = self.attn_fc_dropout(self.fc(x))
        
        return x




class SubsetAttention(nn.Module):
    """Attention that can attend to subsets of queries or keys/values."""
    
    def __init__(
        self,
        dim,
        head_dim,
        qkv_bias=False,
        attn_dropout=0.0,
        fc_bias=False,
        fc_dropout=0.0,
        use_qk_norm=False
    ):
        """
        Args:
            dim: Input dimension
            head_dim: Dimension of each attention head
            qkv_bias: Whether to use bias in QKV projection
            attn_dropout: Dropout probability for attention weights
            fc_bias: Whether to use bias in output projection
            fc_dropout: Dropout probability for output projection
            use_qk_norm: Whether to use Q-K normalization
        We use flash attention V2 for efficiency.
        """
        super().__init__()
        assert dim % head_dim == 0, f"Token dimension {dim} should be divisible by head dimension {head_dim}"
        
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.attn_dropout = attn_dropout
        self.use_qk_norm = use_qk_norm

        # Projections
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.fc = nn.Linear(dim, dim, bias=fc_bias)
        self.attn_fc_dropout = nn.Dropout(fc_dropout)
        
        # Optional Q-K normalization
        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x, subset_kv_size=None, subset_q_size=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            subset_kv_size: If provided, only attend to tokens after this index in KV
            subset_q_size: If provided, only compute attention for queries up to this index
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Only one subset parameter can be provided
        assert not (subset_kv_size is not None and subset_q_size is not None), \
            "Only one of subset_kv_size or subset_q_size can be provided"

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        q, k, v = (rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim) for t in (q, k, v))
        
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        dropout_p = self.attn_dropout if self.training else 0.0
        
        # Handle subset attention cases
        if subset_kv_size is not None and subset_kv_size < k.shape[1]:
            # Attend to subset of key/value tokens
            k_subset = k[:, subset_kv_size:, :, :].contiguous()
            v_subset = v[:, subset_kv_size:, :, :].contiguous()
            
            x = self._compute_attention(q, k_subset, v_subset, dropout_p)
        elif subset_q_size is not None and subset_q_size < q.shape[1]:
            # Only compute attention for subset of query tokens
            q_subset = q[:, :subset_q_size, :, :].contiguous()
            
            x = self._compute_attention(q_subset, k, v, dropout_p)
        else:
            # Regular attention for all tokens
            x = self._compute_attention(q, k, v, dropout_p)
        
        x = rearrange(x, "b l nh dh -> b l (nh dh)")

        # Final projection
        x = self.attn_fc_dropout(self.fc(x))
        
        return x
    
    def _compute_attention(self, q, k, v, dropout_p):
        """
        
        Args:
            q: Query tensor (batch, seq_len_q, nheads, head_dim)
            k: Key tensor (batch, seq_len_kv, nheads, head_dim)
            v: Value tensor (batch, seq_len_kv, nheads, head_dim)
            dropout_p: Dropout probability
            
        Returns:
            Output tensor (batch, seq_len_q, nheads, head_dim)
        """
        if FLASH_ATTN_AVAILABLE and q.is_cuda:
            # Flash Attention
            return flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=None,
                causal=False,
            )
        elif XFORMERS_AVAILABLE:
            # Fallback to xformers
            return xops.memory_efficient_attention(
                q, k, v,
                attn_bias=None,
                p=dropout_p,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
        else:
            # Fallback to PyTorch native scaled_dot_product_attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False,
            )
            return x.transpose(1, 2)




class QK_Norm_TransformerBlock(nn.Module):
    """
    Standard transformer block with pre-normalization architecture.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L95-L113
    """

    def __init__(
        self,
        dim,
        head_dim,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        use_qk_norm=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, bias=ln_bias)
        self.attn = QK_Norm_SelfAttention(
            dim=dim,
            head_dim=head_dim,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
        )

        self.norm2 = nn.LayerNorm(dim, bias=ln_bias)
        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=mlp_bias,
            dropout=mlp_dropout,
        )


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ConditionalPredictorBlock(QK_Norm_TransformerBlock):
    """AdaLN-modulated Transformer block for conditional prediction.
    
    """
    def __init__(self, dim, head_dim, cond_dim=None, mlp_ratio=4, use_qk_norm=True):
        super().__init__(dim=dim, head_dim=head_dim, mlp_ratio=mlp_ratio, use_qk_norm=use_qk_norm)
        cond_dim = cond_dim or dim
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, cond):
        """
        Args:
        """
        if cond.dim() == 3:
            cond = cond.mean(dim=1)  # [B, L_cond, cond_dim] -> [B, cond_dim]
        modulation = self.adaLN_modulation(cond)  # [B, 6*D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        # unsqueeze for broadcast: [B, D] -> [B, 1, D]
        shift_msa = shift_msa.unsqueeze(1)
        scale_msa = scale_msa.unsqueeze(1)
        gate_msa = gate_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)
        
        x = x + gate_msa * self.attn(self._modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(self._modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class GNTViewAttention(nn.Module):
    """
    GNT-style View Transformer Module for cross-attention between target and source views.
    Reference: 'Is Attention All NeRF Needs?', CVPR 2023
    
    Uses flash attention for efficiency when available.
    """
    def __init__(
        self,
        dim,
        head_dim=64,
        n_heads=1,
        qkv_bias=False,
        fc_bias=True,
        attn_dropout=0.0,
        fc_dropout=0.0,
        use_qk_norm=True,
        mlp_ratio=4,
    ):
        """
        Args:
            dim: Input/output dimension
            head_dim: Dimension per attention head (ignored if n_heads is set)
            n_heads: Number of attention heads (overrides head_dim if set)
            qkv_bias: Whether to use bias in Q/K/V projections
            fc_bias: Whether to use bias in output projection
            attn_dropout: Dropout probability for attention weights
            fc_dropout: Dropout probability for output projection
            use_qk_norm: Whether to use Q-K RMS normalization
            mlp_ratio: Multiplier for FFN hidden dimension
        """
        super().__init__()
        
        if n_heads is not None:
            head_dim = dim // n_heads
        
        assert dim % head_dim == 0, f"Token dimension {dim} should be divisible by head dimension {head_dim}"
        
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.attn_dropout = attn_dropout
        self.use_qk_norm = use_qk_norm
        
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.fc = nn.Linear(dim, dim, bias=fc_bias)
        self.attn_fc_dropout = nn.Dropout(fc_dropout)
        
        # Optional Q-K normalization
        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
        
        # LayerNorm for attention output
        self.norm = nn.LayerNorm(dim, bias=False)
        
        # Feed Forward Network
        hidden_dim = int(dim * mlp_ratio)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(fc_dropout),
            nn.Linear(hidden_dim, dim, bias=False),
        )
        self.norm_ff = nn.LayerNorm(dim, bias=False)
    
    def forward(self, query, key, value):
        """
        Cross-attention: query attends to key/value.
        
        Args:
            
        Returns:
            Output tensor of shape [Batch, N_Query, Dim]
        """
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        
        # Reshape to multi-head format: (b, l, d) -> (b, l, nh, dh)
        q = rearrange(q, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
        k = rearrange(k, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
        v = rearrange(v, "b l (nh dh) -> b l nh dh", dh=self.head_dim)
        
        # Apply qk normalization if enabled
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Compute attention using flash attention or fallbacks
        dropout_p = self.attn_dropout if self.training else 0.0
        x = self._compute_attention(q, k, v, dropout_p)
        
        # Reshape back: (b, l, nh, dh) -> (b, l, d)
        x = rearrange(x, "b l nh dh -> b l (nh dh)")
        x = self.attn_fc_dropout(self.fc(x))
        
        # Residual Connection + Norm (add attention output to query)
        x = self.norm(query + x)
        
        # Feed Forward + Residual
        x = self.norm_ff(x + self.feed_forward(x))
        
        return x
    
    def _compute_attention(self, q, k, v, dropout_p):
        """
        
        Args:
            q: Query tensor (batch, seq_len_q, nheads, head_dim)
            k: Key tensor (batch, seq_len_kv, nheads, head_dim)
            v: Value tensor (batch, seq_len_kv, nheads, head_dim)
            dropout_p: Dropout probability
            
        Returns:
            Output tensor (batch, seq_len_q, nheads, head_dim)
        """
        if FLASH_ATTN_AVAILABLE:
            return flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=None,
                causal=False,
            )
        elif XFORMERS_AVAILABLE:
            # Fallback to xformers
            return xops.memory_efficient_attention(
                q, k, v,
                attn_bias=None,
                p=dropout_p,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
            )
        else:
            # Fallback to PyTorch native scaled_dot_product_attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False,
            )
            return x.transpose(1, 2)

def _init_weights_layerwise(module, weight_init_std):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=weight_init_std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=weight_init_std)
