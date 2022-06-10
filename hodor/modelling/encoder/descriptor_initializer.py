from einops import rearrange
from torch import Tensor
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgPoolingInitializer(nn.Module):
    GRID_DIMS = {
        1:  (1, 1),
        2:  (1, 2),
        4:  (2, 2),
        9:  (3, 3),
        12: (3, 4),
        16: (4, 4)
    }

    def __init__(self, num_bg_queries: int, **kwargs):
        super().__init__()
        self.num_bg_queries = num_bg_queries
        self.grid_dims = self.GRID_DIMS[num_bg_queries]
        self.fg_thresh = kwargs.get("fg_thresh", 0.5)
        self.bg_thresh = kwargs.get("bg_thresh", 0.5)

    def get_fold_params(self, height: int, width: int) -> Dict[str, Any]:
        grid_h, grid_w = self.grid_dims
        assert height % grid_h == 0 and width % grid_w == 0, f"Grid dims: {self.grid_dims}, tensor dims: ({height}, {width})"
        patch_h, patch_w = height // grid_h, width // grid_w

        return {
            "kernel_size": (patch_h, patch_w),
            "stride": (patch_h, patch_w),
            "padding": 0
        }

    def unfold(self, x: Tensor) -> Tensor:
        """
        Reshapes tensor into list of patches
        :param x: tensor fo shape [B, C, H, W]
        :return: tensor of shape [B, C, patch_sz, num_patches]
        """
        fold_params = self.get_fold_params(*x.shape[-2:])
        patch_sz = np.prod(fold_params["kernel_size"])

        x = F.unfold(x, **fold_params)  # [B, C * patch_sz, num_patches]
        assert x.size(2) == self.num_bg_queries
        return rearrange(x, "B (C patch_sz) num_patches -> B C patch_sz num_patches", patch_sz=patch_sz)

    @torch.no_grad()
    def get_patch_masks(self, batch_sz: int, height: int, width: int, device: torch.cuda.device) -> Tensor:
        """
        Returns binary masks for the patches given the image feature dims
        :param batch_sz:
        :param height:
        :param width:
        :param device
        :return: tensor of shape [batch_sz, num_patches, height, width]
        """
        num_patches = np.prod(self.grid_dims)
        patch_masks = torch.zeros(batch_sz, num_patches, height, width, dtype=torch.float32, device=device)

        grid_h, grid_w = self.grid_dims
        patch_h, patch_w = height // grid_h, width // grid_w

        for y in range(grid_h):
            start_y = y * patch_h
            if y == grid_h - 1:
                end_y = height
            else:
                end_y = (y + 1) * patch_h

            for x in range(grid_w):
                start_x = x * patch_w
                if x == grid_w - 1:
                    end_x = width
                else:
                    end_x = (x + 1) * patch_w

                patch_id = (y * grid_w) + x

                patch_masks[:, patch_id, start_y:end_y, start_x:end_x] = 1.0

        return patch_masks

    def forward(self, fmap: Tensor, fg_mask: Tensor, bg_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward method
        :param fmap: tensor of shape [B, C, H, W]
        :param fg_mask: tensor of shape [B, I, H, W] with values in [0,1]
        :param bg_mask: tensor of shape [B, Qb, H, W] with values in [0,1]
        :return: tuple of tensors of shape [B, I, C] and [B, Qb, C]
        """
        # bg_mask = self.unfold(bg_mask)  # [B, 1, grid_patch_size, Qb]
        # fmap_unfolded = self.unfold(fmap)  # [B, C, grid_patch_size, Qb]

        ch = fmap.size(1)
        fmap = rearrange(fmap, "B C H W -> B (H W) C")
        fg_mask = rearrange(fg_mask, "B I H W -> B I (H W)")
        bg_mask = rearrange(bg_mask, "B Qb H W -> B Qb (H W)")

        fg_init, bg_init = [], []

        for fmap_ps, fg_mask_ps, bg_mask_ps in zip(fmap, fg_mask, bg_mask):

            fg_init.append([])
            for fg_mask_ps_i in fg_mask_ps:
                fg_pt_coords = (fg_mask_ps_i > self.fg_thresh).nonzero(as_tuple=False).squeeze(1)  # [N]

                if fg_pt_coords.numel() == 0:
                    fg_pt_coords = fg_mask_ps_i.argmax()[None]  # [1]

                fg_init[-1].append(fmap_ps[fg_pt_coords].mean(0))

            fg_init[-1] = torch.stack(fg_init[-1], 0)

            bg_init.append([])
            for bg_mask_ps_i in bg_mask_ps:
                bg_pt_coords = (bg_mask_ps_i > self.bg_thresh).nonzero(as_tuple=False).squeeze(1)  # [N]

                if bg_pt_coords.numel() == 0:
                    bg_init[-1].append(torch.zeros(ch, dtype=fmap.dtype, device=fmap.device))
                else:
                    bg_init[-1].append(fmap_ps[bg_pt_coords].mean(0))

            bg_init[-1] = torch.stack(bg_init[-1], 0)

        return torch.stack(fg_init, 0), torch.stack(bg_init, 0)