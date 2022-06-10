from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class SegmentationMask(object):
    def __init__(self, mask_tensor: torch.Tensor, ds_scale: int, val_type: str):
        assert mask_tensor.ndim == 4, f"Given tensor has {mask_tensor.ndim} dimensions"
        assert mask_tensor.dtype == torch.float32, f"Given tensor has dtype {mask_tensor.dtype}"

        self.t = mask_tensor
        self.ds_scale: int = ds_scale

        assert val_type in ("logits", "probs"), f"Invalid val_type: '{val_type}'"
        self.type: str = val_type

    def detach_(self):
        self.t = self.t.detach()

    def cuda(self):
        return self.__class__(self.t.cuda(), self.ds_scale, self.type)

    def cpu(self):
        return self.__class__(self.t.cpu(), self.ds_scale, self.type)

    def resize(self, target_scale: int):
        assert target_scale in (1, 2, 4, 8, 16, 32), f"Invalid target scale: {target_scale}"

        if target_scale == self.ds_scale:
            resized = self.t.clone()
            ds_scale = self.ds_scale
        else:
            scale_factor = self.ds_scale / float(target_scale)

            resized = F.interpolate(self.t, scale_factor=scale_factor, mode='bilinear', align_corners=False,
                                    recompute_scale_factor=False)

            ds_scale = int(self.ds_scale / scale_factor)

        return self.__class__(resized, ds_scale, self.type)

    def convert(self, target_type: str, eps: Optional[float] = 1e-6):
        assert target_type in ("logits", "probs")

        if target_type == "logits" and self.type == "probs":
            t = torch.logit(self.t, eps=eps)
        elif target_type == "probs" and self.type == "logits":
            t = torch.softmax(self.t, 1)
        else:
            t = self.t.clone()

        return self.__class__(t, self.ds_scale, target_type)

    def copy(self, detach=False):
        copy = self.__class__(self.t.clone(), self.ds_scale, self.type)
        if detach:
            copy.t = copy.t.detach()
        return copy

    def to_logits(self, eps=1e-6):
        return self.convert("logits", eps=eps)

    def to_probs(self):
        return self.convert("probs")

    def get_bg_mask(self):
        assert self.type == "probs"
        return 1. - self.t.sum(1)

    def visualize(self, batch_idx):
        assert self.type == "probs"
        masks = self.t[batch_idx]

        h = masks.size(1)
        v_line = np.full((h, 2), 255, np.uint8)

        vis_masks = []
        for iid, mask_i in enumerate(masks.unbind(0), 1):
            mask_i = (mask_i > 0.5).byte().cpu().numpy()
            vis_masks.append(mask_i * 255)
            vis_masks.append(v_line)

        vis_masks = np.stack(vis_masks, axis=1)

        cv2.namedWindow("Mask Vis", cv2.WINDOW_NORMAL)
        cv2.imshow("Mask Vis", vis_masks)
        cv2.waitKey(0)
