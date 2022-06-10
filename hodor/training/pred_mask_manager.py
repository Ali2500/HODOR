from einops import repeat
from typing import Dict, List, Any, Union
from torch import Tensor
from hodor.modelling.segmentation_mask import SegmentationMask

import torch


@torch.no_grad()
def condense_mask(expanded_mask: Tensor):
    # expanded_mask: [B, expanded_dim, *], dtype=bool
    # return shape: [B, *], dtype=long
    # NOTE: does not account for multiple channels having a True value for the same pixel.
    fg_mask = torch.any(expanded_mask, 1)  # [B, *]
    condensed_mask = expanded_mask.byte().argmax(1) + 1  # [B, *]
    return torch.where(fg_mask, condensed_mask, torch.zeros_like(condensed_mask))


class PredMaskManager:
    _COUNT_VALID = 0
    _COUNT_TOTAL = 0

    def __init__(self):
        self.instance_presence_map: Tensor = None  # [B, T, I]
        self.gt_masks: Tensor = None  # [B, T, H, W]

        self._pred_masks_container: List[Tensor] = []  # entry: [B, 1+I, H, W]
        self._gt_masks_container: List[Union[Tensor, None]] = []  # entry: [B, I, H, W]
        self._frame_ids: List[int] = []
        self._is_present: Tensor = torch.zeros(0)  # [B, I]

        self._current_cycle_presence_history = []
        self._all_cycles_presence_history = []

    @property
    def batch_sz(self):
        return self.gt_masks.size(0)

    @property
    def num_instances(self):
        return self.instance_presence_map.size(2)

    @property
    def masked_out_ratio(self):
        return 1. - (float(PredMaskManager._COUNT_VALID) / float(PredMaskManager._COUNT_TOTAL))

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def is_gt_available(self, frame_id: int):
        return frame_id < self.gt_masks.size(1)

    def start_new_cycle(self): 
        if self._current_cycle_presence_history:
            self._all_cycles_presence_history.append(self._current_cycle_presence_history)
            self._current_cycle_presence_history = []

        self._is_present = torch.ones(self.batch_sz, self.num_instances, dtype=torch.bool)

    def advance_cycle(self, src_frame_ids: List[int], tgt_frame_id: int):
        presence_map = self.instance_presence_map[src_frame_ids]  # [T', B, I]
        presence_map = torch.any(presence_map, 0)  # [B, I]
        self._is_present = torch.logical_and(self._is_present.to(presence_map.device), presence_map)
        
        self._current_cycle_presence_history.append((tgt_frame_id, self._is_present))

    def add_prediction(self, mask: SegmentationMask, src_frame_ids: List[int], tgt_frame_id: int):
        self.advance_cycle(src_frame_ids, tgt_frame_id)

        self._pred_masks_container.append(mask.resize(1).t)
        self._frame_ids.append(tgt_frame_id)

        if self.is_gt_available(tgt_frame_id):
            with torch.no_grad():
                gt_masks_tgt_frame = self.gt_masks[:, tgt_frame_id]  # [B, I, H, W]
                # zero out ground truth for instances which are no longer trackable
                gt_masks_tgt_frame = torch.where(self._is_present[:, :, None, None],
                                                 gt_masks_tgt_frame, 
                                                 torch.zeros_like(gt_masks_tgt_frame))

                self._gt_masks_container.append(gt_masks_tgt_frame)

                PredMaskManager._COUNT_VALID += self._is_present.sum(dtype=torch.long).item()
                PredMaskManager._COUNT_TOTAL += self._is_present.numel()
        else:
            self._gt_masks_container.append(None)

    def get_pred_gt_pairs(self, frame_id: int, condense_gt_mask: bool):
        assert frame_id in self._frame_ids, f"No masks exist for t={frame_id}"

        idxes = [i for i, t in enumerate(self._frame_ids) if t == frame_id]

        pred_masks = torch.stack([self._pred_masks_container[i] for i in idxes], 2)  # [B, 1+I, N, H, W]
        gt_masks = torch.stack([self._gt_masks_container[i] for i in idxes], 2)  # [B, I, N, H, W]

        # sanity check:
        assert pred_masks.size(1) - 1 == self.num_instances, f"Shape mismatch: {pred_masks.shape}, num_instances"

        if condense_gt_mask:
            gt_masks = condense_mask(gt_masks)  # [B, N, H, W]
            assert list(gt_masks.size()) == [self.batch_sz, len(idxes)] + list(pred_masks.shape[-2:]), \
                f"Shape mismatch: {gt_masks.shape}, [{[self.batch_sz, len(idxes)] + list(pred_masks.shape[-2:])}]"

        return pred_masks, gt_masks

    def get_all_pred_gt_pairs(self, condense_gt_mask: bool):
        idxes = [i for i in range(len(self._pred_masks_container)) if self._gt_masks_container[i] is not None]

        pred_masks = torch.stack([self._pred_masks_container[i] for i in idxes], 2)  # [B, 1+I, N, H, W]
        gt_masks = torch.stack([self._gt_masks_container[i] for i in idxes], 2)  # [B, I, N, H, W]

        # sanity check:
        assert pred_masks.size(1) - 1 == self.num_instances, f"Shape mismatch: {pred_masks.shape}, num_instances"

        if condense_gt_mask:
            gt_masks = condense_mask(gt_masks)  # [B, N, H, W]
            assert list(gt_masks.size()) == [self.batch_sz, len(idxes)] + list(pred_masks.shape[-2:]), \
                f"Shape mismatch: {gt_masks.shape}, [{[self.batch_sz, len(idxes)] + list(pred_masks.shape[-2:])}]"

        return pred_masks, gt_masks

    @classmethod
    def from_tensors(cls,
                     instance_presence_map: Tensor,
                     gt_ref_mask: Tensor,
                     gt_other_masks: Union[Tensor, None]):
        """Create an instance from tensors

        Args:
            instance_presence_map (Tensor): boolean tensor of shape [batch_sz, num_frames, num_instances]
            gt_ref_mask (Tensor): Ground truth mask for reference frame of shape [batch_sz, height, width] with pixel
                                  values denoting instance ID
            gt_other_masks (Union[Tensor, None]): Optional ground truth mask of shape [batch_sz, num_frames-1, height, width].
                                                  This will not be available when training with cyclic consistency or on
                                                  single images.

        Returns:
            [type]: Object instance
        """

        obj = cls()
        obj.instance_presence_map = repeat(instance_presence_map, "B T I -> T B I")
        num_instances = instance_presence_map.size(2)

        if gt_other_masks is None:  # no intermediate frame supervision available or single frame training
            gt_masks = gt_ref_mask.unsqueeze(1)  # [B, 1, H, W]
        else:
            gt_masks = torch.cat((gt_ref_mask.unsqueeze(1), gt_other_masks), 1)  # [B, T, H, W]

        # split gt masks into separate binary masks per instance
        obj.gt_masks = torch.stack([gt_masks == instance_id for instance_id in range(1, num_instances + 1)], 2)  # [B, T/1, I, H, W]

        return obj
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        obj = cls()
        for k in obj.__dict__.keys():
            setattr(obj, k, d[k])

        return obj