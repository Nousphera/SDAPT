# --------------------------------------------------------
# References:
# DinoV2: https://github.com/facebookresearch/dinov2
# Dino: https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
# TIMM: https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py
# --------------------------------------------------------

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn

from .lora_layers_util import LoRALinearLayer


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_proj, self.k_proj, self.v_proj = None, None, None  # Should be initialized after loading state_dict
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_lora(
        self,
        lora_rank=4,
        include_attn_key=False,
        include_proj=False,
    ):
        dim = self.qkv.in_features
        weight = self.qkv.weight.data
        q_weight, k_weight, v_weight = torch.chunk(weight, 3)  # (out_features, in_features)

        self.q_proj = LoRALinearLayer(dim, dim, r=lora_rank, bias=self.qkv.bias is not None)
        self.q_proj.weight.data.copy_(q_weight)

        self.k_proj = (
            LoRALinearLayer(dim, dim, r=lora_rank, bias=self.qkv.bias is not None)
            if include_attn_key
            else nn.Linear(dim, dim, bias=self.qkv.bias is not None)
        )
        self.k_proj.weight.data.copy_(k_weight)

        self.v_proj = LoRALinearLayer(dim, dim, r=lora_rank, bias=self.qkv.bias is not None)
        self.v_proj.weight.data.copy_(v_weight)

        if self.qkv.bias is not None:
            bias = self.qkv.bias.data
            q_bias, k_bias, v_bias = torch.chunk(bias, 3)  # (out_features)
            self.q_proj.bias.data.copy_(q_bias)
            self.k_proj.bias.data.copy_(k_bias)
            self.v_proj.bias.data.copy_(v_bias)

        del self.qkv

        if include_proj:
            dim = self.proj.in_features
            self.proj = LoRALinearLayer(dim, dim, r=lora_rank, bias=self.proj.bias is not None)

    def deinit_lora(self, delete_separate_proj=False):
        # assert is_merged(self)
        if self.q_proj is None or self.k_proj is None or self.v_proj is None:
            return

        dim = self.k_proj.in_features

        qkv_bias = self.q_proj.bias is not None and self.k_proj.bias is not None and self.v_proj.bias is not None
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        q_weight, k_weight, v_weight = self.q_proj.weight.data, self.k_proj.weight.data, self.v_proj.weight.data
        weight = torch.cat((q_weight, k_weight, v_weight))
        self.qkv.weight.data.copy_(weight)

        if qkv_bias:
            q_bias, k_bias, v_bias = self.q_proj.bias.data, self.k_proj.bias.data, self.v_proj.bias.data
            bias = torch.cat((q_bias, k_bias, v_bias))
            self.qkv.bias.data.copy_(bias)

        if delete_separate_proj:
            self.q_proj, self.k_proj, self.v_proj = None, None, None

    def forward(self, x: Tensor) -> Tensor:
        assert hasattr(self, "q_proj") and hasattr(self, "v_proj") and hasattr(self, "k_proj")
        B, N, C = x.shape
        if self.q_proj is not None:
            qkv = torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)  # B, N, 3C
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        if self.q_proj is not None:
            qkv = torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)  # B, N, 3C
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
