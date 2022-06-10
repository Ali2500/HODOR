from torch import Tensor
from typing import Dict

from hodor.modelling.backbone.swin import build_swin_fpn

import torch.nn as nn


class _SwinTransformerFPN(nn.Module):
    def __init__(self, pretrained: bool, variant: str):
        super().__init__()
        self.bb_fpn = build_swin_fpn(variant, pretrained)
        self.norm_layers = nn.ModuleList([nn.GroupNorm(16, 256) for _ in range(5)])

        # self.is_3d = False
        self.output_scales = [4, 8, 16, 32]

    @property
    def num_output_scales(self):
        return len(self.output_scales)

    def forward(self, x: Tensor) -> Dict[int, Tensor]:
        fmaps = self.bb_fpn(x)
        return {scale: norm(f) for scale, norm, f in zip(self.output_scales, self.norm_layers, fmaps)}


class SwinTransformerFPNTiny(_SwinTransformerFPN):
    def __init__(self, pretrained: bool, n_dims: int):
        assert n_dims == 256
        super(SwinTransformerFPNTiny, self).__init__(pretrained, variant="tiny")


class SwinTransformerFPNSmall(_SwinTransformerFPN):
    def __init__(self, pretrained: bool, n_dims: int):
        assert n_dims == 256
        super(SwinTransformerFPNSmall, self).__init__(pretrained, variant="small")


class SwinTransformerFPNBase(_SwinTransformerFPN):
    def __init__(self, pretrained: bool, n_dims: int):
        assert n_dims == 256
        super(SwinTransformerFPNBase, self).__init__(pretrained, variant="base")
