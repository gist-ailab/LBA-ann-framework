# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


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
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, return_attn=False, return_qkv=False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        
        if return_qkv:
            return q, k, v
        
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Add those 2 lines
        if return_attn:
            return attn
        
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, return_attn=False, return_qkv=False) -> Tensor:
        # if return_attn:
        #     return super().forward(x, return_attn)
        
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            # Change this line
            # return super().forward(x)
            return super().forward(x, return_attn, return_qkv)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        q, k, v = unbind(qkv, 2)
        
        if return_qkv:
            return q, k, v
        
        # if return_attn:
        #     q = q.swapaxes(1,2)
        #     k = k.swapaxes(1,2)
        #     v = v.swapaxes(1,2)
 
        #     scale = 1 / q.shape[-1] ** 0.5
        #     q = q * scale
        #     attn = q @ k.transpose(-2, -1)
        #     # if attn_bias is not None:
        #     #     attn = attn + attn_bias
        #     # attn = attn.softmax(-1)
        #     # attn = self.attn_drop(attn)
            
        #     return attn
        
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        
        if return_attn:
            # scale = 1 / q.shape[-1] ** 0.5
            # q = q * scale
            
            attn = x.permute(0, 2, 1, 3) @ v.permute(0, 2, 3, 1)
            # attn = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
            attn = attn.softmax(-1)
            return attn
        
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

