from copy import deepcopy
from torch import Tensor
from typing import Dict
from einops import rearrange, repeat
from hodor.modelling.encoder.layer import EncoderLayer
from hodor.modelling.encoder.descriptor_initializer import AvgPoolingInitializer
from hodor.modelling.position_embeddings import SinosuidalPositionEmbeddings

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_heads, n_dims: int, n_layers: int, n_bg_queries: int, pre_normalize: bool, pos_encodings: bool):
        super().__init__()

        self.query_initializer = AvgPoolingInitializer(n_bg_queries)

        attn_layer = EncoderLayer(
            n_dims, n_heads=n_heads, n_hidden_dims=1024, dropout=0.1, pre_normalize=pre_normalize
        )

        self.attn_layers = nn.ModuleList([deepcopy(attn_layer) for _ in range(n_layers)])

        self.pos_encoding_gen = SinosuidalPositionEmbeddings(n_dims // 2, normalize=True)
        self.use_pos_encodings = pos_encodings

        self.norm = nn.LayerNorm(n_dims) if pre_normalize else nn.Identity()

        self.register_parameter("fg_query_embed", nn.Parameter(torch.zeros(n_dims), True))
        self.register_parameter("bg_query_embed", nn.Parameter(torch.zeros(n_bg_queries, n_dims), True))

        self.num_bg_queries = n_bg_queries
        self.mask_scale = 8
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.fg_query_embed)
        nn.init.normal_(self.bg_query_embed)

    def get_fmap(self, fmaps: Dict[int, Tensor]):
        return fmaps[self.mask_scale]

    def forward(self, fmaps: Dict[int, Tensor], fg_mask: Tensor, bg_mask: Tensor) -> Dict[str, Tensor]:
        """
        Forward method
        :param fmaps: Tensor of shape [B, C, H, W] containing reference frame features.
        :param fg_mask: Tensor of shape [B, I, H, W] with values in range [0, 1]
        :param bg_mask: Tensor of shape [B, Qb, H, W] with values in range [0, 1]
        :return:
        """
        assert fg_mask.dtype in (torch.float16, torch.float32)
        assert not torch.any(fg_mask > (1. + 1e-6)) and not torch.any(fg_mask < -1e-6)
        assert not torch.any(bg_mask > (1. + 1e-6)) and not torch.any(bg_mask < -1e-6)

        fmap = self.get_fmap(fmaps)

        bs, num_inst = fg_mask.shape[:2]

        with torch.no_grad():
            if self.use_pos_encodings:
                pos_encoding = self.pos_encoding_gen(fmap)
                pos_encoding = rearrange(pos_encoding, "B C H W -> B (H W) C")
            else:
                pos_encoding = None

            fg_queries, bg_queries = self.query_initializer(fmap=fmap, fg_mask=fg_mask, bg_mask=bg_mask)

        fmap = rearrange(fmap, "B C H W -> B (H W) C")

        all_queries = torch.cat((bg_queries, fg_queries), 1)  # [B, Qb+I, C]

        bg_query_pos = repeat(self.bg_query_embed, "Qb C -> B Qb C", B=bs)
        fg_query_pos = repeat(self.fg_query_embed, "C -> B I C", B=bs, I=num_inst)
        all_query_pos = torch.cat((bg_query_pos, fg_query_pos), 1)  # [B, Qb+I, C]

        bg_mask = rearrange(bg_mask, "B Qb H W -> B Qb (H W)")
        fg_mask = rearrange(fg_mask, "B I H W -> B I (H W)")
        all_masks = torch.cat((bg_mask, fg_mask), 1)  # [B, Qb+I, H*W]

        for layer in self.attn_layers:
            all_queries = layer(query=all_queries, kv=fmap, pos_key=pos_encoding, kv_mask=all_masks,
                                pos_query=all_query_pos)

        all_queries = self.norm(all_queries)

        bg_queries, fg_queries = all_queries.split((self.num_bg_queries, num_inst), 1)

        return {
            "fg_queries": fg_queries,
            "bg_queries": bg_queries
        }
