from collections import defaultdict
from einops import repeat
from torch import Tensor
from typing import List, Tuple, Dict, Union, Any
from hodor.modelling.segmentation_mask import SegmentationMask
from hodor.modelling.model_base import ModelBase

import torch
import torch.nn.functional as F


class InferenceModel(ModelBase):
    def __init__(self):
        super(InferenceModel, self).__init__()

        self.temporal_window = 0
        self.backbone_batch_size = 16

        # temporary caches
        self._fg_query_history: Dict[int, Dict[int, Tensor]] = defaultdict(dict)  # tensor: [C]
        self._bg_query_history: Dict[int, Tensor] = dict()  # tensor: [Qb, C]

    def clear_caches(self):
        self._fg_query_history.clear()
        self._bg_query_history.clear()

    def get_current_num_instances(self):
        return len(set(sum([list(v.keys()) for v in self._fg_query_history.values()], [])))

    def get_query_history(self, frame_id: int):
        """
        Fetch history of instance embeddings
        :param frame_id: int
        :return: Tuple of 5 outputs:
                 (1) foreground instance queries as tensor of shape [1, I, T, C]
                 (2) background instance queries as tensor of shape [1, Qb, T, C]
                 (3) the instance IDs of the foreground objects
                 (4) Mask which is true for invalid foreground queries as tensor of shape [1, I, T]
                 (4) Mask which is true for invalid background queries as tensor of shape [1, Qb, T]
                 (5) list of relative frame indices as a tensor of shape [T]
        """
        t_past = max(0, frame_id - self.temporal_window)
        t_range = list(range(t_past, frame_id + 1))  # include current frame as well

        frame_indices = torch.arange(frame_id - t_past, 1, -1).cuda().unsqueeze(0)  # [1, T]
        default = torch.zeros(256, dtype=torch.float32).cuda()
        instance_ids_in_history = sorted(list(set(sum([list(self._fg_query_history[t].keys()) for t in t_range], []))))

        fg_queries = torch.stack([
            torch.stack([
                self._fg_query_history[t].get(iid, default)
                for iid in instance_ids_in_history
            ], 0)
            for t in t_range
        ], 1)  # [I, T, C]

        default = torch.zeros(self.num_bg_queries, 256, dtype=torch.float32).cuda()

        bg_queries = torch.stack(
            [self._bg_query_history.get(t, default) for t in t_range], 1
        )  # [Qb, T, C]

        invalid_fg_query_mask = torch.stack([
            torch.as_tensor([iid not in self._fg_query_history[t] for iid in instance_ids_in_history], dtype=torch.bool)
            for t in t_range
        ], 1)  # [I, T]

        invalid_bg_query_mask = torch.as_tensor([t not in self._bg_query_history for t in t_range], dtype=torch.bool).cuda()  # [T]
        invalid_bg_query_mask = repeat(invalid_bg_query_mask, "T -> 1 Qb T", Qb=self.num_bg_queries)

        return (
            fg_queries.unsqueeze(0),
            bg_queries.unsqueeze(0),
            instance_ids_in_history,
            invalid_fg_query_mask.unsqueeze(0),
            invalid_bg_query_mask,
            frame_indices
        )

    def reset_bg_masks(self, masks: SegmentationMask) -> Tuple[Tensor, Tensor]:
        num_bg = self.num_catch_all_queries + self.num_bg_queries
        _, fg_masks = masks.t.split((num_bg, masks.t.size(1) - num_bg), 1)  # [B, I, H, W]

        batch_sz, _, height, width = fg_masks.shape
        bg_patch_masks = self.encoder.query_initializer.get_patch_masks(
            batch_sz, height, width, device=masks.t.device
        )  # [B, Qb, H, W]

        soft_bg_mask = 1. - fg_masks.sum(1, keepdim=True)
        if not torch.all(torch.logical_and(soft_bg_mask >= -1e-6, soft_bg_mask <= (1.0 + 1e-6))):
            tmp = fg_masks.sum(1)  # [B, H, W]
            n_offending_vals = torch.logical_or(tmp < 0.0, tmp > 1.0).sum(dtype=torch.long).item()
            print(f"[WARN] Background mask values do not lie in the range [0, 1]. Offending values in foreground mask:"
                  f"min={tmp.min().item()}, max={tmp.max().item()}, no. of offending values={n_offending_vals}")

        soft_bg_masks = bg_patch_masks * soft_bg_mask

        return soft_bg_masks, fg_masks

    def reverse_resize_and_padding(self, masks: Tensor, resized_dims: Tuple[int, int], orig_dims: Tuple[int, int]) -> Tensor:
        """
        Resizes a given set of masks by reversing the padding and resizing ops applied in the data loader.
        :param masks: tensor of mask probabilities of shape [N, H, W]
        :param resized_dims: tuple of (height, width). This is the size of the resized image before padding
        :param orig_dims: tuple of (height, width). This is the original size of the image.
        :return: tensor of shape [N, H, W] of type long (here H, W are the 'orig_dims')
        """
        resized_h, resized_w = resized_dims
        orig_h, orig_w = orig_dims
        current_h, current_w = masks.shape[1:]

        padding_h, padding_w = current_h - resized_h, current_w - resized_w
        masks = F.pad(masks[None], (0, -padding_w, 0, -padding_h), mode='constant', value=0)
        return F.interpolate(masks, (orig_h, orig_w), mode='bilinear', align_corners=False)[0]  # [I, H, W]

    def apply_instance_map(self, masks: Tensor, instance_id_map: List[int], includes_bg: bool) -> Tensor:
        """
        Converts a set of instance masks into a single mask with non-overlapping entries corresponding to the given
        instance IDs
        :param masks: tensor of shape [Qb+I, H, W] (first Qb masks are background) or [I, H, W]
        :param instance_id_map: mapping of instance IDs of length [I]
        :param includes_bg: bool
        :return: tensor of shape [H, W]
        """
        if includes_bg:
            assert masks.size(0) == self.num_bg_queries + self.num_catch_all_queries + len(instance_id_map), \
                f"{masks.size(0)} =/= {self.num_bg_queries + self.num_catch_all_queries + len(instance_id_map)}"

            # all background and catch all predictions should be mapped to 0 in the final mask
            instance_id_map = [0 for _ in range(self.num_bg_queries + self.num_catch_all_queries)] + instance_id_map
        else:
            assert masks.size(0) == len(instance_id_map), f"{masks.size()} =/= {len(instance_id_map)}"
            h, w = masks.shape[1:]
            masks = torch.cat((torch.zeros(1, h, w).to(masks), masks), 0)
            instance_id_map = [0] + instance_id_map

        masks = masks.argmax(0)  # [H, W]

        mapped_mask = torch.zeros_like(masks)
        for ii, mapped_ii in enumerate(instance_id_map):
            mapped_mask = torch.where(masks == ii, torch.full_like(mapped_mask, mapped_ii), mapped_mask)

        return mapped_mask

    def override_pred_mask_with_gt(self,
                                   pred_mask: SegmentationMask,
                                   pred_mask_instance_id_map: List[int],
                                   gt_mask: Union[Dict[int, Tensor], None]):

        pred_mask_out = pred_mask.resize(1).to_probs()
        pred_mask_internal = pred_mask.to_probs().resize(1)

        if gt_mask is None:
            return pred_mask_internal.resize(self.encoder_mask_scale), pred_mask_out

        num_bg = self.num_bg_queries + self.num_catch_all_queries

        def process(pm):
            bg_masks = pm.t[0, :num_bg]  # [Qc + Qb, H, W]
            fg_masks = pm.t[0, num_bg:]  # [I, H, W]
            reverse_id_map = {iid: index for index, iid in enumerate(pred_mask_instance_id_map)}

            for iid, mask_iid in gt_mask.items():
                assert mask_iid.shape == fg_masks.shape[1:], f"Shape mismatch: {mask_iid.shape}, {fg_masks.shape}"
                idx = reverse_id_map[iid]

                # zero out bg and fg masks at locations where the ground truth mask is positive
                bg_masks = torch.where(mask_iid[None], torch.zeros_like(bg_masks), bg_masks)
                fg_masks = torch.where(mask_iid[None], torch.zeros_like(fg_masks), fg_masks)

                # set the mask of the new instance equal to the ground truth mask
                fg_masks[idx] = mask_iid

            pm.t = torch.cat((bg_masks, fg_masks), 0).unsqueeze(0)
            return pm

        pred_mask_internal = process(pred_mask_internal)
        pred_mask_out = process(pred_mask_out)

        return pred_mask_internal.resize(self.encoder_mask_scale), pred_mask_out

    @torch.no_grad()
    def forward(self, frames: Tensor, ref_masks_container: Dict[int, List[Tuple[int, Tensor]]], resized_dims: Tuple[int, int],
                orig_dims: Tuple[int, int]) -> List[Tensor]:
        """
        Forward method
        :param frames: tensor of shape [T, C, H, W]
        :param ref_masks_container: Dict with keys denoting frame ID, and values being a list of tuples, each with 2 elements:
                          (1) The instance ID
                          (2) The reference mask for that instance as a tensor of shape [H, W]
        :param resized_dims: (height, width) tuple
        :param orig_dims: (height, width) tuple
        :param show_vis:
        :returns segmentation masks as tensor of shape [T, H, W] of type long
        """
        self.clear_caches()

        fmaps_all = {scale: [] for scale in self.feature_extractor.output_scales}
        seq_len, _, height, width = frames.shape

        for t in range(0, seq_len, self.backbone_batch_size):
            end_t = min(t + self.backbone_batch_size, frames.size(0))
            fmaps_batch = self.feature_extractor(frames[t:end_t].cuda())

            for scale, f in fmaps_batch.items():
                fmaps_all[scale].append(f.cpu())

        for scale in fmaps_all.keys():
            fmaps_all[scale] = torch.cat(fmaps_all[scale], 0)  # [T, C, H, W]

        input_h, input_w = [int(fmaps_all[4].shape[i] * 4) for i in (-2, -1)]

        final_masks = []
        # ref_instance_ids = []

        for t in range(seq_len):
            current_num_instances = self.get_current_num_instances()

            if current_num_instances == 0 and t not in ref_masks_container:
                # no pre-existing instances and no new instances in frame 't'. Just write out an all-zeros mask and
                # move on.
                final_masks.append(torch.zeros(input_h, input_w, dtype=torch.long).cuda())
                continue

            fmaps_t = {scale: f[t][None].cuda() for scale, f in fmaps_all.items()}

            # (1) If any new instances are starting from this frame, we need to generate latent embeddings for them.
            if t in ref_masks_container:
                # print("new instances found")
                output = self.ref_masks_to_queries(
                    ref_masks_container=ref_masks_container[t],
                    fmaps=fmaps_t
                )

                for iid, fg_query_iid in zip(output["instance_ids"], output["fg_queries"].unbind(0)):
                    self._fg_query_history[t][iid] = fg_query_iid

                if current_num_instances == 0:
                    self._bg_query_history[t] = output["bg_queries"]

                ref_mask_t = output["ref_fg_mask"]
                # ref_mask_expanded_t = output["expanded_ref_mask"]
            else:
                ref_mask_t = None

            if current_num_instances > 0:
                # (2) If there are any pre-existing instances, those need to be propagated to the current timestep
                fg_queries, bg_queries, instance_ids_t, fg_query_padding_mask, bg_query_padding_mask, frame_indices = \
                    self.get_query_history(t)

                # print(fg_queries.shape, bg_queries.shape, fg_query_padding_mask.shape, bg_query_padding_mask.shape)
                # print(t, fg_queries.shape, bg_queries.shape, fg_query_padding_mask.tolist())

                fq_m_output = self.decoder(
                    fg_queries=fg_queries, bg_queries=bg_queries, fmaps=fmaps_t,
                    fg_query_embed=self.pos_embed_fg_descriptors, bg_query_embed=self.pos_embed_bg_descriptors,
                    fg_query_mask=fg_query_padding_mask.cuda(), bg_query_mask=bg_query_padding_mask.cuda()
                )

                pred_masks_t = fq_m_output["pred_mask"]  # [1, Qb+I H, W]. First Qb entries in dim=0 contains bg logits
                # pred_masks_t = pred_masks_t.to_probs()

                # Here we have a strange situation. For instances with reference (ground truth) masks in the current
                # frame, we have actually predicted a mask for them in the current frame as well. This predicted mask
                # will have some errors in it, so just for these instances, we will replace the mask with the available
                # ground truth mask.
                pred_masks_t, output_masks_t = self.override_pred_mask_with_gt(
                    pred_masks_t, instance_ids_t, ref_mask_t
                )

                final_mask_t = self.reverse_resize_and_padding(output_masks_t.t.squeeze(0), resized_dims, orig_dims)
                final_mask_t = self.apply_instance_map(final_mask_t, instance_ids_t, includes_bg=True)
                final_masks.append(final_mask_t)

                # (3) For the instances propagated in step (2), we need to generate object embeddings.
                if t == seq_len - 1:
                    continue  # no need to create queries for the last frame in the sequence

                # reset the background masks
                bg_mask, fg_mask = self.reset_bg_masks(pred_masks_t)
                fm_q_output = self.encoder(
                    fmaps=fmaps_t, fg_mask=fg_mask, bg_mask=bg_mask
                )

                fg_queries = fm_q_output["fg_queries"].squeeze(0)
                bg_queries = fm_q_output["bg_queries"].squeeze(0)

                self._bg_query_history[t] = bg_queries

                for iid, fg_query_iid in zip(instance_ids_t, fg_queries.unbind(0)):
                    self._fg_query_history[t][iid] = fg_query_iid

            else:
                instance_ids, instance_masks = zip(*[(iid, mask) for iid, mask in ref_mask_t.items()])
                instance_masks = torch.stack(instance_masks, 0)  # [I, H, W]
                instance_masks = self.reverse_resize_and_padding(instance_masks.float(), resized_dims, orig_dims)
                final_masks.append(self.apply_instance_map(instance_masks, list(instance_ids), includes_bg=False))

        return final_masks

    def ref_masks_to_queries(self,
                             ref_masks_container: List[Tuple[int, Tensor]],
                             fmaps: Dict[int, Tensor]) -> Dict[str, Any]:

        # get instance IDs and instance mask for the instances with reference frames at timestep 't'
        instance_ids, fg_ref_masks = zip(*[ref_masks_container[i] for i in range(len(ref_masks_container))])
        # print(instance_ids)
        fg_ref_masks = torch.stack(fg_ref_masks, 0)  # [I, H, W]

        # combine the foreground and background masks
        h, w = fg_ref_masks.shape[-2:]
        bg_ref_mask_t = torch.where(torch.all(fg_ref_masks < 0.5, 0),
                                    torch.ones(h, w, dtype=torch.float32),
                                    torch.zeros(h, w, dtype=torch.float32))

        split_bg_masks = self.encoder.query_initializer.get_patch_masks(
            1, h, w, device=bg_ref_mask_t.device
        ).squeeze(0)

        # set fg pixel locations to zero in the bg masks
        split_bg_masks = torch.where(
            bg_ref_mask_t.unsqueeze(0) > 0.5, split_bg_masks, torch.zeros_like(split_bg_masks)
        )

        ref_masks = torch.cat((split_bg_masks, fg_ref_masks), 0).cuda()  # [Qb+I, H, W]

        # this is the mask that will be given to FM->Q later
        pred_masks = SegmentationMask(ref_masks.unsqueeze(0), 1, "probs").resize(self.encoder_mask_scale)

        # detect all-zeros masks (very small instances)
        valid_masks_flag = torch.any(pred_masks.t[0, self.num_bg_queries:].flatten(1) > 0.5, 1)  # [Qb+I]
        n_invalid_masks = valid_masks_flag.numel() - valid_masks_flag.sum().item()

        if n_invalid_masks > 0:
            print(f"Number of invalid reference masks: {n_invalid_masks}")
            bg_masks, fg_masks = pred_masks.t.split((self.num_bg_queries, len(instance_ids)), 1)

            fg_masks = fg_masks.squeeze(0)[valid_masks_flag].unsqueeze(0)
            instance_ids = [
                instance_ids[i] for i in range(len(instance_ids))
                if valid_masks_flag[i].item()
            ]

            pred_masks.t = torch.cat((bg_masks, fg_masks), 1)
            fg_ref_masks = fg_ref_masks[torch.as_tensor(valid_masks_flag, dtype=torch.bool)]

        # separate gt foreground masks
        assert fg_ref_masks.size(0) == len(instance_ids), f"{fg_ref_masks.shape}, {len(instance_ids)}"
        fg_ref_masks = {
            iid: mask.cuda() for iid, mask in zip(instance_ids, fg_ref_masks)
        }

        # run FM->Q to get queries for the reference masks
        fm_q_output = self.encoder(
            fmaps=fmaps, fg_mask=pred_masks.t[:, self.num_bg_queries:], bg_mask=pred_masks.t[:, :self.num_bg_queries]
        )

        # add queries to cache
        fg_queries = fm_q_output["fg_queries"].squeeze(0)  # [I, C]
        bg_queries = fm_q_output["bg_queries"].squeeze(0)  # [Qb, C]

        return {
            "fg_queries": fg_queries,
            "bg_queries": bg_queries,
            "ref_fg_mask": fg_ref_masks,
            "pred_mask": pred_masks,
            "instance_ids": instance_ids
        }

    def dump_queries_to_file(self, output_path):
        with open(output_path, "wb") as fh:
            torch.save({
                "fg_queries": self._fg_query_history,
                "bg_queries": self._bg_query_history
            }, fh)


def build_inference_model():
    return InferenceModel()
