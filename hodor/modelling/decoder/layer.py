from einops import rearrange, repeat
from torch import Tensor
from typing import Union, Optional

from hodor.modelling.decoder.attention import MultiheadAttention
from torchvision.ops.deform_conv import DeformConv2d

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformConvBlock(nn.Module):
    def __init__(self, n_dims, n_offset_grps):
        super().__init__()
        self.conv_offsets = nn.Conv2d(n_dims, 3 * 3 * 2 * n_offset_grps, 3, padding=1)
        self.conv_deform = DeformConv2d(n_dims, n_dims, 3, padding=1, groups=n_offset_grps)

    def forward(self, x: Tensor):
        offsets = self.conv_offsets(x)
        return self.conv_deform(x, offsets)


class DecoderLayer(nn.Module):
    def __init__(self, n_dims: int, n_heads: int, n_hidden_dims: int, dropout: float = 0.1,
                 pre_normalize: bool = False):
        super().__init__()

        self.self_attn_conv_substitute = nn.Sequential(
            DeformConvBlock(n_dims, 1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(16, n_dims)
        )

        self.k_proj_fg = nn.Linear(n_dims, n_dims)
        self.k_proj_bg = nn.Linear(n_dims, n_dims)

        self.v_proj_fg = nn.Linear(n_dims, n_dims)
        self.v_proj_bg = nn.Linear(n_dims, n_dims)

        if pre_normalize:
            self.norm1 = nn.Identity()
        else:
            self.norm1 = nn.LayerNorm(n_dims)

        self.cross_attn = MultiheadAttention(n_dims, n_heads, dropout=dropout, k_proj=False, v_proj=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(n_dims)
        self.fc1 = nn.Linear(n_dims, n_hidden_dims)
        self.dropout2 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(n_hidden_dims, n_dims)
        self.dropout3 = nn.Dropout(dropout)

        self.pre_normalize = pre_normalize
        self.cyclic_consistency = False

    @staticmethod
    def add_pos_embedding(x: Tensor, pos: Union[Tensor, None]) -> Tensor:
        if pos is None:
            return x
        else:
            assert torch.is_tensor(pos)
            return x + pos

    def forward(self, fg_queries: Tensor, bg_queries: Tensor, fmap: Tensor, pos_fmap: Optional[Tensor] = None,
                pos_fg_queries: Optional[Tensor] = None, pos_bg_queries: Optional[Tensor] = None,
                fg_query_mask: Optional[Tensor] = None, bg_query_mask: Optional[Tensor] = None):
        """
        Forward method
        :param fg_queries: queries of shape [B, I, T, C]
        :param bg_queries: queries of shape [B, Qb, T, C]
        :param fmap: feature map of shape [B, C, H, W]
        :param pos_fmap: optional query encodings of shape [B, C, H, W]
        :param pos_fg_queries: queries positional encodings of shape [B, I, T, C]
        :param pos_bg_queries: queries positional encodings of shape [B, Qb, T, C]
        :param fg_query_mask: optional mask to remove some queries from consideration. Shape [B, I, T]
        :param bg_query_mask: optional mask to remove some queries from consideration. Shape [B, Qb, T]
        :param time_indices: optional tensor of shape [B, Q]

        :return: updated feature map of shape [B, C, H, w]
        """
        fmap = fmap + self.self_attn_conv_substitute(fmap.contiguous())
        bs, c, h, w = fmap.shape

        fmap = rearrange(fmap, "B C H W -> B (H W) C")

        if pos_fmap is not None:
            pos_fmap = rearrange(pos_fmap, "B C H W -> B (H W) C")

        fmap = self.forward_cross_attn(
            fg_queries, bg_queries, fmap,
            pos_fmap=pos_fmap, pos_fg_queries=pos_fg_queries, pos_bg_queries=pos_bg_queries,
            fg_query_mask=fg_query_mask, bg_query_mask=bg_query_mask
        )
        assert fmap.shape == (bs, h*w, c)

        return rearrange(fmap, "B (H W) C -> B C H W", H=h, W=w).contiguous()

    def forward_cross_attn(self, fg_queries: Tensor,
                           bg_queries: Tensor,
                           fmap: Tensor,
                           pos_fmap: Optional[Tensor] = None,
                           pos_fg_queries: Optional[Tensor] = None,
                           pos_bg_queries: Optional[Tensor] = None,
                           fg_query_mask: Optional[Tensor] = None,
                           bg_query_mask: Optional[Tensor] = None) -> Tensor:

        fg_queries = rearrange(fg_queries, "B I T C -> B (I T) C")
        bg_queries = rearrange(bg_queries, "B Qb T C -> B (Qb T) C")

        if pos_fg_queries is not None:
            pos_fg_queries = rearrange(pos_fg_queries, "B I T C -> B (I T) C")

        if pos_bg_queries is not None:
            pos_bg_queries = rearrange(pos_bg_queries, "B Qb T C -> B (Qb T) C")

        key_fg = self.k_proj_fg(self.add_pos_embedding(fg_queries, pos_fg_queries))  # [B, I*T, C]
        key_bg = self.k_proj_bg(self.add_pos_embedding(bg_queries, pos_bg_queries))  # [B, Qb*T, C]
        key = torch.cat((key_fg, key_bg), 1)  # [B, (I*T) + (Qb*T), C]

        value_fg = self.v_proj_fg(fg_queries)
        value_bg = self.v_proj_bg(bg_queries)
        value = torch.cat((value_fg, value_bg), 1)  # [B, (I*T) + (Qb*T), C]

        # if time_indices is not None:
        #     time_indices = repeat(time_indices, "B T -> B (I T)", I=num_inst+1)

        query_mask = None

        if fg_query_mask is not None:
            assert fg_query_mask.dtype == torch.bool
            assert bg_query_mask.size(2) == fg_query_mask.size(2), \
                f"Shape mismatch: {fg_query_mask.shape}, {bg_query_mask.shape}"

            fg_query_mask = rearrange(fg_query_mask, "B I T -> B (I T)")
            bg_query_mask = rearrange(bg_query_mask, "B Qb T -> B (Qb T)")
            query_mask = torch.cat((fg_query_mask, bg_query_mask), 1)  # [B, (I*T) + (Qb*T)]

        # embedding_ids = None
        # if not self.training and self.cyclic_consistency:
        #     embedding_ids = self.get_embedding_ids(bs, num_inst, num_bg, num_timesteps, query_mask)

        res = self.cross_attn(
            query=self.add_pos_embedding(fmap, pos_fmap),
            key=key,
            value=value,
            activation="softmax",
            key_mask=query_mask
        )

        fmap = self.norm1(fmap + self.dropout1(res))

        res = fmap
        res = self.dropout2(F.relu(self.fc1(res)))
        res = self.dropout3(self.fc2(res))

        if self.pre_normalize:
            return fmap + self.norm2(res)
        else:
            return self.norm2(fmap + res)

    # def get_embedding_ids(self, batch_sz: int, num_inst: int, num_bg: int, num_timesteps: int,
    #                       query_mask: Optional[Tensor] = None) -> Tensor:
    #     fg_ids = torch.arange(1, num_inst + 1, dtype=torch.long, device=query_mask.device)
    #     fg_ids = repeat(fg_ids, "NumInst -> Bs NumInst T", Bs=batch_sz, T=num_timesteps).flatten(1, 2)
    #
    #     bg_ids = torch.full((batch_sz, num_bg * num_timesteps), fill_value=num_inst + 1, dtype=torch.long,
    #                         device=query_mask.device)
    #
    #     combined_ids = torch.cat((fg_ids, bg_ids), 1)  # [B, I*T + Qb*T]
    #     # breakpoint()
    #     assert query_mask.shape == combined_ids.shape, f"Shape mismatch: {query_mask.shape}, {combined_ids.shape}"
    #
    #     if query_mask is None:
    #         return combined_ids
    #     else:
    #         return torch.where(query_mask, torch.zeros_like(combined_ids), combined_ids)


# class AttentionBlockCombinedQueriesDeformableConv(AttentionBlockCombinedQueriesDilatedConv):
#     def __init__(self, n_dims: int, n_heads: int, n_hidden_dims: int, dropout: float = 0.1, n_offset_grps: int = 1,
#                  alibi_te: bool = False, pre_normalize: bool = False):
#         super(AttentionBlockCombinedQueriesDeformableConv, self).__init__(
#             n_dims, n_heads, n_hidden_dims, dropout, alibi_te=alibi_te, pre_normalize=pre_normalize
#         )
#
#         class DeformConvBlock(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.conv_offsets = nn.Conv2d(n_dims, 3*3*2 * n_offset_grps, 3, padding=1)
#                 self.conv_deform = DeformConv2d(n_dims, n_dims, 3, padding=1, groups=n_offset_grps)
#
#             def forward(self, x: Tensor):
#                 offsets = self.conv_offsets(x)
#                 return self.conv_deform(x, offsets)
#
#         self.self_attn_conv_substitute = nn.Sequential(
#             DeformConvBlock(),
#             nn.ReLU(inplace=True),
#             nn.GroupNorm(16, n_dims)
#         )
