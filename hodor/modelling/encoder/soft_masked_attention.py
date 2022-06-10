from einops import rearrange, repeat
from typing import Optional

import math
import torch
import torch.nn as nn
from torch import Tensor


class SoftMaskedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, learnable_scaling_factors=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dims = embed_dim // num_heads
        assert self.head_dims * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        if learnable_scaling_factors:
            self.scaling_factors = nn.Parameter(torch.as_tensor(self.get_default_scaling_factors(), dtype=torch.float32))
        else:
            self.register_buffer("scaling_factors", torch.as_tensor(self.get_default_scaling_factors(), dtype=torch.float32))

    def get_default_scaling_factors(self):
        assert self.num_heads == 8
        return [32, 32, 16, 16, 8, 8, 4, 4]

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor,
                key_padding_mask: Optional[Tensor] = None, return_attn_weights: Optional[bool] = False):
        """
        Forward method
        :param query: [bs, num_instances, embed_dim]
        :param key: [bs, src_len, embed_dim]
        :param value: [bs, src_len, embed_dim]
        :param mask: tensor of shape [bs, num_instances, src_len] with values in range [0, 1]
        :param key_padding_mask: optional tensor of shape [bs, src_len] of type bool. Value should be true for padded
        locations and false for valid locations.
        :param return_attn_weights: if True, attention weights will also be returned as a tensor of shape
        [bs, num_heads, num_instances, src_len]
        :return: tensor of shape [bs, num_instances, embed_dim]
        """
        assert key.shape == value.shape, f"Shape mismatch: {key.shape}, {value.shape}"
        assert query.shape[:2] == mask.shape[:2], f"Shape mismatch: {query.shape}, {mask.shape}"

        batch_sz = query.size(0)
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = rearrange(query, "bs num_inst (num_heads head_dims) -> (bs num_heads) num_inst head_dims",
                          num_heads=self.num_heads, head_dims=self.head_dims)
        key = rearrange(key, "bs src_len (num_heads head_dims) -> (bs num_heads) head_dims src_len",
                        num_heads=self.num_heads, head_dims=self.head_dims)
        value = rearrange(value, "bs src_len (num_heads head_dims) -> (bs num_heads) src_len head_dims",
                          num_heads=self.num_heads, head_dims=self.head_dims)

        query = query / float(math.sqrt(self.head_dims))

        attn_wts = torch.bmm(query, key)  # [bs * num_heads, num_inst, src_len]

        mask = repeat(1. - mask, "bs num_inst src_len -> bs num_heads num_inst src_len", num_heads=self.num_heads)
        mask = (mask * self.scaling_factors[None, :, None, None]).flatten(0, 1)  # [bs * num_heads, num_inst, src_len]

        attn_wts = attn_wts - mask

        if key_padding_mask is not None:
            key_padding_mask = repeat(key_padding_mask, "bs src_len -> (bs num_heads) num_inst src_len",
                                      num_heads=self.num_heads, num_inst=query.size(1))
            attn_wts = torch.where(key_padding_mask, torch.full_like(attn_wts, float("-inf")), attn_wts)

        attn_wts = attn_wts.softmax(-1)
        attn_wts = self.dropout(attn_wts)

        attn_output = torch.bmm(attn_wts, value)
        attn_output = rearrange(attn_output, "(bs num_heads) num_inst head_dims -> bs num_inst (num_heads head_dims)",
                                num_heads=self.num_heads, head_dims=self.head_dims)

        if return_attn_weights:
            attn_wts = rearrange(attn_wts, "(bs num_heads) num_inst src_len -> bs num_heads num_inst src_len", bs=batch_sz)
            return self.out_proj(attn_output), attn_wts
        else:
            return self.out_proj(attn_output)
