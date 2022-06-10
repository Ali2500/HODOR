from typing import Optional, Union
from torch import Tensor
from hodor.modelling.encoder.soft_masked_attention import SoftMaskedAttention

import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, n_dims: int, n_heads: int, n_hidden_dims: int, dropout: float = 0.1, pre_normalize: bool = False):
        super().__init__()

        self.norm1 = nn.LayerNorm(n_dims)
        self.self_attn = nn.MultiheadAttention(n_dims, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(n_dims)
        self.cross_attn = SoftMaskedAttention(n_dims, n_heads, dropout=dropout, learnable_scaling_factors=True)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(n_dims)

        self.ffn = nn.Sequential(
            nn.Linear(n_dims, n_hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden_dims, n_dims),
            nn.Dropout(dropout)
        )

        self.pre_normalize = pre_normalize

    @staticmethod
    def add_pos_embedding(x: Tensor, pos: Union[Tensor, None]) -> Tensor:
        if pos is None:
            return x
        else:
            assert torch.is_tensor(pos)
            return x + pos

    def forward(self, query: Tensor, kv: Tensor, kv_mask: Optional[Tensor] = None, pos_key: Optional[Tensor] = None,
                pos_query: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward method
        :param query: tensor of shape [B, I, C]
        :param kv: tensor of shape [B, S, C]
        :param kv_mask: tensor of shape [B, I, S]
        :param pos_key: optional tensor of shape [B, S, C]
        :param pos_query: optional tensor of shape [B, I, C]
        :param key_padding_mask: optional tensor of shape [B, S] of type bool. Value should be true for padded
        locations and false for valid locations.
        :return: tensor of shape [B, T, C]
        """
        pre_norm1, pre_norm2, pre_norm3, post_norm1, post_norm2, post_norm3 = [lambda x: x for _ in range(6)]

        if self.pre_normalize:
            pre_norm1 = self.norm1
            pre_norm2 = self.norm2
            pre_norm3 = self.norm3
        else:
            post_norm1 = self.norm1
            post_norm2 = self.norm2
            post_norm3 = self.norm3

        res = pre_norm1(query)

        res = res.transpose(0, 1)
        if pos_query is not None:
            pos_query = pos_query.transpose(0, 1)

        kq = self.add_pos_embedding(res, pos_query)

        res = self.self_attn(query=kq, key=kq, value=res)[0]
        res = res.transpose(0, 1)

        query = post_norm1(query + self.dropout1(res))

        res = pre_norm2(query)
        res = self.cross_attn(query=res,
                              key=self.add_pos_embedding(kv, pos_key),
                              value=kv,
                              mask=kv_mask,
                              key_padding_mask=key_padding_mask)
        query = post_norm2(query + self.dropout2(res))

        res = pre_norm3(query)
        res = self.ffn(res)
        return post_norm3(query + res)


# class CrossAndSelfAttentionBlockMultiInstanceOffsetMask(CrossAndSelfAttentionBlockMultiInstance):
#     def __init__(self, n_dims: int, n_heads: int, n_hidden_dims: int, dropout: float = 0.1, pre_normalize: bool = False,
#                  learnable_scaling_factors: bool = True):
#
#         super(CrossAndSelfAttentionBlockMultiInstanceOffsetMask, self).__init__(
#             n_dims=n_dims, n_heads=n_heads, n_hidden_dims=n_hidden_dims, dropout=dropout, pre_normalize=pre_normalize
#         )
#
#         self.cross_attn = MaskOffsetMultiheadAttention(n_dims, n_heads, dropout=dropout,
#                                                        learnable_scaling_factors=learnable_scaling_factors)
