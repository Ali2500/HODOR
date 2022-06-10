from einops import rearrange, repeat
from torch import Tensor
from typing import Optional

import math
import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self, n_dims, n_heads, dropout=0.0, q_proj=True, k_proj=True, v_proj=True):
        super().__init__()

        self.dims_per_head = n_dims // n_heads
        assert self.dims_per_head * n_heads == n_dims
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Identity()
        if q_proj:
            self.q_proj = nn.Linear(n_dims, n_dims)

        self.k_proj = nn.Identity()
        if k_proj:
            self.k_proj = nn.Linear(n_dims, n_dims)

        self.v_proj = nn.Identity()
        if v_proj:
            self.v_proj = nn.Linear(n_dims, n_dims)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, activation: str, key_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward method
        :param query: tensor of shape [B, T, C]
        :param key: tensor of shape [B, S, C]
        :param value: tensor of shape [B, S, C]
        :param activation: string
        :param key_mask: optional mask for the key tensor of shape [B, S] and type bool. Should be true for keys which
        have to be masked out.
        :return: tensor of shape [B, T, C]
        """
        bs, t = query.shape[:2]
        s = key.shape[1]
        assert key.shape == value.shape, f"Shape mismatch: {key.shape}, {value.shape}"
        assert key.size(0) == query.size(0) and key.size(2) == query.size(2), f"Shape mismatch: {query.shape}, {key.shape}"

        query = rearrange(self.q_proj(query), "BS TgtLen (NumHeads DimsPerHead) -> (BS NumHeads) TgtLen DimsPerHead",
                          NumHeads=self.n_heads, DimsPerHead=self.dims_per_head)

        key = rearrange(self.k_proj(key), "BS SrcLen (NumHeads DimsPerHead) -> (BS NumHeads) DimsPerHead SrcLen",
                        NumHeads=self.n_heads, DimsPerHead=self.dims_per_head)

        value = rearrange(self.v_proj(value),  "BS SrcLen (NumHeads DimsPerHead) -> (BS NumHeads) SrcLen DimsPerHead",
                          NumHeads=self.n_heads, DimsPerHead=self.dims_per_head)

        query = query / float(math.sqrt(self.dims_per_head))
        attn_wts = torch.bmm(query, key)  # [BS * NumHeads, TgtLen, SrcLen]
        assert list(attn_wts.shape) == [bs * self.n_heads, t, s]

        attn_wts = self.dropout(attn_wts)

        if key_mask is not None:
            key_mask = repeat(key_mask, "BS SrcLen -> BS NumHeads TgtLen SrcLen", NumHeads=self.n_heads, TgtLen=t)
            key_mask = rearrange(key_mask, "BS NumHeads TgtLen SrcLen -> (BS NumHeads) TgtLen SrcLen")
            attn_wts = torch.where(key_mask, torch.full_like(attn_wts, float("-inf")), attn_wts)

        if activation == "sigmoid":
            attn_wts = attn_wts.sigmoid()
        elif activation == "softmax":
            attn_wts = attn_wts.softmax(2)
        elif activation != "none":
            raise ValueError("Invalid activation '{}'".format(activation))

        attn = torch.bmm(attn_wts, value)  # [B*H, T, C']

        attn = rearrange(attn, "(BS NumHeads) TgtLen DimsPerHead -> BS TgtLen (NumHeads DimsPerHead)",
                         BS=bs, NumHeads=self.n_heads, DimsPerHead=self.dims_per_head).contiguous()

        assert list(attn.shape) == [bs, t, self.dims_per_head * self.n_heads]
        return attn
