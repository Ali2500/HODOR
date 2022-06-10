from copy import deepcopy
from einops import rearrange, repeat
from torch import Tensor
from typing import List, Union, Optional, Dict

from hodor.modelling.decoder.layer import DecoderLayer
from hodor.modelling.position_embeddings import SinosuidalPositionEmbeddings
from hodor.modelling.segmentation_mask import SegmentationMask

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Sequential):
    def __init__(self, in_planes, out_planes, *args, **kwargs):
        super(Conv2d, self).__init__(
            nn.Conv2d(in_planes, out_planes, *args, **kwargs, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, out_planes)
        )


class Decoder(nn.Module):
    def __init__(self, n_dims, n_layers, n_heads, n_catch_all_queries, pre_normalize, pos_encodings):
        super().__init__()

        attn_layer = DecoderLayer(n_dims, n_heads, n_dims, pre_normalize=pre_normalize)
        self.attn_layers = nn.ModuleList([deepcopy(attn_layer) for _ in range(n_layers)])

        self.conv_4x_reduce = Conv2d(n_dims, 64, 1)
        self.conv_4x = Conv2d(256, 256, 3, padding=1)

        if pre_normalize:
            self.norm_fmap = nn.GroupNorm(1, n_dims)
        else:
            self.norm_fmap = nn.Identity()

        self.instance_query_proj = nn.Linear(n_dims, n_dims)
        self.bg_query_proj = nn.Linear(n_dims, n_dims)

        self.fmap_proj_4x = Conv2d(n_dims, n_dims, 1)

        assert n_catch_all_queries == 1
        self.catch_all_conv = nn.Sequential(
            nn.Conv2d(n_dims, n_dims // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, n_dims // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_dims // 2, 1, 1, bias=False)
        )

        if pos_encodings:
            self.pos_encoding_gen = SinosuidalPositionEmbeddings(n_dims // 2, normalize=True)
        else:
            self.pos_encoding_gen = lambda x: None

    def generate_per_instance_masks(self, instance_queries: Tensor, fmap: Tensor,
                                    instance_queries_invalid_mask: Union[None, Tensor]) -> Tensor:
        """
        Generate per instance masks using matrix product
        :param instance_queries: tensor of shape [B, I, T, C]
        :param fmap: tensor of shape [B, C, H, W]
        :param instance_queries_invalid_mask: tensor of shape [B, I, T] or None
        :return: tensor of shape [B, 1+I, H, W]
        """
        bs, num_inst = instance_queries.shape[:2]
        c, h, w = fmap.shape[-3:]

        catch_all_bg = self.catch_all_conv(fmap)  # [B, 1, H, W]

        fmap = repeat(fmap, "B C H W -> (B I) C H W", I=num_inst)
        fmap = rearrange(fmap, "BI C H W -> BI C (H W)")

        instance_queries = rearrange(instance_queries, "B I T C -> (B I) T C")

        prod = torch.bmm(instance_queries, fmap) / float(math.sqrt(c))  # [B*I, T, H*W]

        if instance_queries_invalid_mask is not None:
            instance_queries_invalid_mask = instance_queries_invalid_mask.flatten(0, 1)[:, :, None]  # [B*I, T, 1]
            assert not self.training

            assert instance_queries_invalid_mask.shape[:2] == instance_queries.shape[:2], \
                f"{instance_queries_invalid_mask.shape}, {instance_queries.shape}"

            prod = torch.where(instance_queries_invalid_mask, torch.full_like(prod, float("-inf")), prod)

        prod = rearrange(prod, "(B I) T (H W) -> B I T H W", B=bs, I=num_inst, H=h, W=w)

        prod = prod.max(2)[0]  # [B, I, H, W]
        prod = torch.cat((catch_all_bg, prod), 1)  # [B, 1+I, H, W]

        return prod

    def forward(self, fg_queries: Tensor, bg_queries: Tensor, fmaps: Dict[int, Tensor], fg_query_embed: Tensor,
                bg_query_embed: Tensor, fg_query_mask: Optional[Tensor] = None,
                bg_query_mask: Optional[Tensor] = None):
        """
        Forward method.

        Args:
            fg_queries (Tensor]): Shape [B, I, T, C]
            bg_queries (Tensor): Shape [B, Qb, T, C]
            fmaps (List[Tensor]): List of tensors, each of shape [B, C, Hx, Wx]
            fg_query_embed (Tensor): Shape [C]
            bg_query_embed (Tensor): Shape [Qb, C]
            fg_query_mask (Tensor): Shape [B, I, T]
            bg_query_mask (Tensor): Shape [B, Qb, T]

        Returns:
            Tensor: Segmentation mask of shape [B, Qb+I, H, W]
        """
        f_4x, f_8x = fmaps[4], fmaps[8]

        fmap = f_8x
        bs, num_inst, t = fg_queries.shape[:3]

        fg_query_embed = repeat(fg_query_embed, "C -> B I T C", B=bs, I=num_inst, T=t)
        bg_query_embed = repeat(bg_query_embed, "Qb C -> B Qb T C", B=bs, T=t)

        with torch.no_grad():
            pos_encodings = self.pos_encoding_gen(fmap)

        for layer in self.attn_layers:
            fmap = layer(fg_queries=fg_queries, bg_queries=bg_queries, fmap=fmap, pos_fmap=pos_encodings,
                         pos_fg_queries=fg_query_embed, pos_bg_queries=bg_query_embed,
                         fg_query_mask=fg_query_mask, bg_query_mask=bg_query_mask)

        fmap = self.norm_fmap(fmap)
        fmap = F.interpolate(fmap, scale_factor=2.0, mode='bilinear', align_corners=False)

        f_4x = self.conv_4x_reduce(f_4x)
        fmap = self.partial_channel_add(fmap, f_4x)
        fmap = self.conv_4x(fmap)

        instance_queries = self.instance_query_proj(fg_queries)
        bg_queries = self.bg_query_proj(bg_queries)

        instance_queries = torch.cat((bg_queries, instance_queries), 1)  # [B, Qb+I, T, C]

        if fg_query_mask is not None:
            instance_queries_invalid_mask = torch.cat((bg_query_mask, fg_query_mask), 1)  # [B, Qb+I, T]
        else:
            instance_queries_invalid_mask = None

        masks = self.generate_per_instance_masks(instance_queries, fmap, instance_queries_invalid_mask)  # [B, Qb+I, H, W]

        return {
            "pred_mask": SegmentationMask(masks, 4, "logits")
        }

    @staticmethod
    def partial_channel_add(x: torch.Tensor, y: torch.Tensor):
        """
        Adds two tensors with different number of channels
        :param x: tensor of shape [B, C1, ...]
        :param y: tensor of shape [B, C2, ...]
        :return: tensor of shape [B, max(C1,C2), ...]
        """
        x_dims = [d for i, d in enumerate(x.shape) if i != 1]
        y_dims = [d for i, d in enumerate(x.shape) if i != 1]
        assert x_dims == y_dims, f"Shape mismatch: {x.shape}, {y.shape}"

        if x.size(1) == y.size(1):
            return x + y
        elif x.size(1) < y.size(1):
            x, y = y, x

        x1, x2 = x.split((y.size(1), x.size(1) - y.size(1)), 1)
        x1 = x1 + y
        return torch.cat((x1, x2), 1)
