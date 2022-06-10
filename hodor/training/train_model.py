from torch import Tensor
from typing import Tuple, Union, Dict, Optional

from hodor.config import cfg
from hodor.modelling.segmentation_mask import SegmentationMask
from hodor.modelling.model_base import ModelBase
from hodor.training.pred_mask_manager import PredMaskManager
from hodor.training.cycle_sampler import Cycle

import logging
import torch


class TrainModel(ModelBase):
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.logger = logger

    def separate_masks(self, ref_mask: Tensor, num_instances: int) -> Tensor:
        """
        Separates a multi-instance mask into separate binary masks and the required number of background masks.
        :param ref_mask: tensor of type long of shape [B, H, W]. All batch samples must have the same number of instances.
        :param num_instances
        :return: Tensor of shape [B, Qb+I, H, W] where the first Qb entries along dim=1 represents the background patch masks
        """
        split_bg_masks = self.encoder.query_initializer.get_patch_masks(*ref_mask.shape, device=ref_mask.device)

        # set bg masks to zero at fg pixel locations
        split_bg_masks = torch.where(ref_mask.unsqueeze(1) > 0, torch.zeros_like(split_bg_masks), split_bg_masks)

        separated_masks = torch.stack([ref_mask == iid for iid in list(range(1, num_instances + 1))], 1)  # [B, I, H, W]
        return torch.cat((split_bg_masks, separated_masks), 1)  # [B, Qb + I, H, W]

    @staticmethod
    def split_bg_fg_masks(combined_masks: Union[Tensor, SegmentationMask]) -> Tuple[Tensor, Tensor]:
        if isinstance(combined_masks, SegmentationMask):
            combined_masks = combined_masks.t

        total = combined_masks.size(1)
        return combined_masks.split((cfg.MODEL.NUM_BG_DESCRIPTORS, total - cfg.MODEL.NUM_BG_DESCRIPTORS), 1)

    def merge_bg_masks(self, masks: SegmentationMask):
        assert masks.type == "logits"
        masks = masks.copy()
        total = masks.t.size(1)
        num_bg = cfg.MODEL.NUM_BG_DESCRIPTORS + self.num_catch_all_queries
        bg_masks, fg_masks = masks.t.split((num_bg, total - num_bg), 1)
        bg_masks = bg_masks.max(1, keepdim=True)[0]
        masks.t = torch.cat((bg_masks, fg_masks), 1)
        return masks

    def reset_bg_masks(self, masks: SegmentationMask) -> Tuple[Tensor, Tensor]:
        num_bg = self.num_catch_all_queries + cfg.MODEL.NUM_BG_DESCRIPTORS
        _, fg_masks = masks.t.split((num_bg, masks.t.size(1) - num_bg), 1)  # [B, I, H, W]

        batch_sz, _, height, width = fg_masks.shape
        bg_patch_masks = self.encoder.query_initializer.get_patch_masks(
            batch_sz, height, width, device=masks.t.device
        )  # [B, Qb, H, W]

        soft_bg_mask = 1. - fg_masks.sum(1, keepdim=True)
        if not torch.all(torch.logical_and(soft_bg_mask >= -1e-6, soft_bg_mask <= 1.0 + 1e-6)):
            tmp = fg_masks.sum(1)  # [B, H, W]
            n_offending_vals = torch.logical_or(tmp < 0.0, tmp > 1.0).sum(dtype=torch.long).item()
            print(f"[WARN] Background mask values do not lie in the range [0, 1]. Offending values in foreground mask:"
                  f"min={tmp.min().item()}, max={tmp.max().item()}, no. of offending values={n_offending_vals}")

        soft_bg_masks = bg_patch_masks * soft_bg_mask

        return soft_bg_masks, fg_masks

    def extract_image_features(self, ref_frame: Tensor, other_frames: Union[Tensor, None]) -> Dict[int, Tensor]:
        """
        Extracts backbone features for the video frames
        :param ref_frame: [B, C, H, W]
        :param other_frames: [B, T, C, H, W] or None
        :return: dictionary with keys as scale ints and values being feature maps
        """
        bs = ref_frame.size(0)
        num_other_frames = other_frames.size(1) if torch.is_tensor(other_frames) else 0
        assert num_other_frames == cfg.TRAINING.NUM_OTHER_FRAMES  # sanity check

        if num_other_frames == 0:  # only single image
            fmaps = self.encoder(ref_frame)
            return {scale: f.unsqueeze(1) for scale, f in fmaps.items()}

        images = torch.cat((ref_frame.unsqueeze(1), other_frames), 1).flatten(0, 1)  # [B*(T+1), C, H, W]

        fmaps = self.feature_extractor(images)
        fmaps = {scale: f.view(bs, num_other_frames + 1, *f.shape[1:]) for scale, f in fmaps.items()}

        return fmaps

    def forward(self, ref_frame: Tensor, ref_mask: Tensor, other_frames: Tensor, iter_num: int, num_instances: int,
                pred_masks_manager: PredMaskManager) -> Dict[str, Tensor]:
        """
        Forward method

        :param ref_frame: Reference frame images as tensor of shape [B, C, H, W]
        :param ref_mask: Patch masks for each reference images as tensor of shape [B, H, W]
        :param other_frames: [B, T, C, H, W]
        :param iter_num:
        :param num_instances:
        :param pred_masks_manager:
        :return:
        """
        num_other_frames = other_frames.size(1) if torch.is_tensor(other_frames) else 0

        if num_other_frames == 0:
            cycles = [Cycle.create(forward=[], backward=[{"src": [0], "tgt": 0}], num_other_frames=0)]

        elif num_other_frames == 1:
            cycles = [
                Cycle.create(forward=[], backward=[{"src": [0], "tgt": 0}], num_other_frames=0),
                Cycle.from_string_spec(1, forward="[0]->1;", backward="[1]->0;"),
                Cycle.from_string_spec(1, forward="[0]->1;", backward="[0,1]->0;")
            ]

        elif num_other_frames == 2:
            cycles = [
                Cycle.create(forward=[], backward=[{"src": [0], "tgt": 0}], num_other_frames=0),
                Cycle.from_string_spec(2, forward="[0]->1;", backward="[1]->0;")
            ]

            if iter_num % 2 == 0:
                cycles.append(Cycle.from_string_spec(2, forward="[0]->1; [1]->2;", backward="[2]->1; [1]->0;"))
            else:
                cycles.append(Cycle.from_string_spec(2, forward="[0]->1; [0,1]->2;", backward="[2]->1; [1,2]->0;"))

        else:
            raise NotImplementedError(f"Cycles for {num_other_frames+1} frame sequences not defined")

        assert len(cycles) > 0

        fmaps = self.extract_image_features(ref_frame, other_frames)

        with torch.no_grad():
            ref_gt_masks = self.separate_masks(ref_mask, num_instances)
            ref_gt_masks = SegmentationMask(ref_gt_masks.float(), 1, "probs").resize(self.encoder_mask_scale)

        f_in = {scale: f[:, 0] for scale, f in fmaps.items()}

        bg_masks, fg_masks = self.split_bg_fg_masks(ref_gt_masks)
        ref_encoder_output = self.encoder(fmaps=f_in, fg_mask=fg_masks, bg_mask=bg_masks)

        fg_descriptor_cache = dict()
        bg_descriptor_cache = dict()

        for cycle in cycles:
            pred_masks_manager.start_new_cycle()
            self.run_cycle(fmaps, cycle, ref_encoder_output, fg_descriptor_cache, bg_descriptor_cache,
                           pred_masks_manager)

        pred_masks_manager.start_new_cycle()

        # When training on multiple GPUs using DDP, the model's forward method should only
        # return 'standard' data types such as Tensors, dicts, list, tuples etc. If we return
        # a custom data type then something goes wrong deep inside the DDP API and it throws 
        # strange errors. We therefore convert this instance of `PredMaskManager` to a dict
        # and then recreate the `PredMaskManager` instance afterwards.
        return pred_masks_manager.to_dict()

    def run_cycle(self, fmaps: Dict[int, Tensor], cycle: Cycle, ref_encoder_output: Dict[str, Tensor],
                  fg_descriptor_cache: Dict[str, Tensor], bg_descriptor_cache: Dict[str, Tensor],
                  pred_mask_manager: PredMaskManager):
        fg_descriptors = {0: ref_encoder_output["fg_queries"]}  # [B, I, C]
        bg_descriptors = {0: ref_encoder_output["bg_queries"]}  # [B, Qb, C]

        for src_t, tgt_t, trace, is_forward_step in cycle.steps():
            if is_forward_step:
                assert tgt_t != 0
                # assert requires_grad == (not trace[-2] == "'")  # sanity check

            if trace in fg_descriptor_cache and is_forward_step:
                fg_descriptors[tgt_t] = fg_descriptor_cache[trace]
                bg_descriptors[tgt_t] = bg_descriptor_cache[trace]
                pred_mask_manager.advance_cycle(src_t, tgt_t)
            else:
                # (1) Src queries + Tgt features -> Tgt mask
                src_fg_descs = torch.stack([fg_descriptors[t] for t in src_t], 2)  # [B, I, T, C]
                src_bg_descs = torch.stack([bg_descriptors[t] for t in src_t], 2)  # [B, Qb, T, C]

                f_in = {scale: f[:, tgt_t] for scale, f in fmaps.items()}

                # with profiler.record_function("FQ->M"):
                decoder_output = self.decoder(
                    fmaps=f_in, fg_queries=src_fg_descs, bg_queries=src_bg_descs,
                    fg_query_embed=self.pos_embed_fg_descriptors, bg_query_embed=self.pos_embed_bg_descriptors
                )
                pred_tgt_mask = decoder_output["pred_mask"]

                pred_mask_manager.add_prediction(self.merge_bg_masks(pred_tgt_mask), src_t, tgt_t)

                # (2) Tgt mask + Tgt features -> Tgt queries
                pred_tgt_mask = pred_tgt_mask.to_probs().resize(self.encoder_mask_scale)

                if tgt_t != 0:
                    if cfg.TRAINING.DETACH_INTERMEDIATE_MASKS:
                        pred_tgt_mask.detach_()

                    bg_masks, fg_masks = self.reset_bg_masks(pred_tgt_mask)
                    encoder_output = self.encoder(fmaps=f_in, fg_mask=fg_masks, bg_mask=bg_masks)

                    fg_descriptors[tgt_t] = encoder_output["fg_queries"]
                    bg_descriptors[tgt_t] = encoder_output["bg_queries"]

                    if is_forward_step:
                        fg_descriptor_cache[trace] = encoder_output["fg_queries"]
                        bg_descriptor_cache[trace] = encoder_output["bg_queries"]
