from collections import defaultdict
from typing import List, Dict
from torch.utils.data import Dataset

from hodor.config import cfg
from hodor.data.utils.multi_instance_mask import MultiInstanceMask
from hodor.data.utils.preprocessing import compute_resize_params, scale_and_normalize_images, ToTorchTensor, \
    BatchImageTransform
from hodor.data.vos.video_augmenter import VideoAugmenter
from hodor.data.vos.json_parser import parse_generic_video_dataset, VideoSequence

import imgaug.augmenters as iaa
import numpy as np
import math
import random
import torch
import torch.nn.functional as F


class DenseVideoDataset(Dataset):
    def __init__(self, base_dir: str, json_path: str, name: str, samples_to_create: int, num_instances: int,
                 min_delta_t: int, max_delta_t: int, min_ref_mask_area: int, min_ref_mask_dim: int,
                 apply_color_augmentations: bool):

        super().__init__()

        assert samples_to_create > 0

        self.base_dir = base_dir

        self.num_other_frames = cfg.TRAINING.NUM_OTHER_FRAMES
        self.min_delta_t = min_delta_t
        self.max_delta_t = max_delta_t

        self.min_ref_mask_area = min_ref_mask_area
        self.min_ref_mask_dim = min_ref_mask_dim
        self.dataset_name = name

        sequences, meta_info = parse_generic_video_dataset(base_dir, json_path)

        for seq in sequences:
            seq.filter_objects_by_size(min_ref_mask_area, min_ref_mask_dim)

        # prune very short sequences
        sequences = [
            seq for seq in sequences
            if len(seq) > (self.num_other_frames + self.min_delta_t) and seq.num_obj_annotations > 0
        ]

        self.sequences = {seq.id: seq for seq in sequences}

        assert max_delta_t - min_delta_t >= self.num_other_frames

        self.samples, self.sample_image_dims, self.sample_num_instances = self.create_training_subsequences(
            sequences, samples_to_create, num_instances
        )

        self.np_to_tensor = BatchImageTransform(
            ToTorchTensor(format='CHW')
        )

        if apply_color_augmentations:
            self.color_augmenter = iaa.Sequential([
                iaa.AddToHueAndSaturation(value_hue=(-12, 12), value_saturation=(-12, 12)),
                iaa.LinearContrast(alpha=(0.95, 1.05)),
                iaa.AddToBrightness(add=(-20, 20))
            ])
        else:
            self.color_augmenter = iaa.Identity()

        self.geometry_augmenter = VideoAugmenter(0.6, min_ref_mask_area)

    @staticmethod
    def expand_sample(attribs):
        assert len(attribs) == 4
        return {
            "seq_id": attribs[0],
            "ref_frame": attribs[1],
            "other_frames": attribs[2],
            "ref_instance_ids": attribs[3]
        }

    @staticmethod
    def compress_sample(sample):
        keys = ("seq_id", "ref_frame", "other_frames", "ref_instance_ids")
        return [sample[k] for k in keys]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.expand_sample(self.samples[idx])

        seq: VideoSequence = self.sequences[sample['seq_id']]

        all_frame_idxes = [sample["ref_frame"]] + sample["other_frames"]
        images = seq.load_images(frame_idxes=all_frame_idxes)

        instance_ids = sample["ref_instance_ids"]
        masks = seq.load_masks(all_frame_idxes, instance_ids, absent_instance_behavior="return_zeros_mask")

        combined_masks = []

        for t in range(len(masks)):
            combined_mask_t = np.zeros((seq.height, seq.width), np.int8)

            for idx in range(len(instance_ids)):
                combined_mask_t = np.where(masks[t][idx], idx + 1, combined_mask_t)

            combined_masks.append(combined_mask_t)

        parsed_dict = self.process_clip(images, combined_masks, len(instance_ids))

        ref_image = parsed_dict["images"][0]  # [C, H, W]
        other_images = parsed_dict["images"][1:]  # [T, C, H, W]

        ref_mask = parsed_dict["masks"][0]  # [H, W]
        other_masks = torch.stack(parsed_dict["masks"][1:], 0)  # [T-1, H, W]

        meta = parsed_dict["meta"]
        meta["dataset"] = self.dataset_name
        meta["seq_id"] = seq.id
        meta["frame_idxes"] = all_frame_idxes
        meta["num_instances"] = len(instance_ids)

        return {
            "ref_image": ref_image,
            "ref_mask": ref_mask,
            "other_masks": other_masks,
            "other_images": other_images,
            "instance_presence_map": parsed_dict["instance_presence_map"],
            "meta": meta
        }

    def process_clip(self, images: List[np.ndarray], masks: List[np.ndarray], num_instances: int):
        """Processes a video clip of images and masks

        Args:
            images (List[np.ndarray]): list of images, each of shape [H, W, C]
            masks (List[np.ndarray]): list of masks, each of shape [H, W]
            num_instances: int

        Returns:
            Dict:
        """
        assert self.num_other_frames > 0

        image_height, image_width = images[0].shape[:2]

        # apply random horizontal flip and frame sequence reversal in training mode
        images, masks = self.apply_random_flip(images, masks)

        # apply color augmentation
        augmenter = self.color_augmenter.to_deterministic()
        images = [augmenter.augment_image(image) for image in images]

        instance_presence_map = self.get_mask_presence_map(masks, num_instances)

        # apply geometric augmentations
        images, masks, invalid_pts_mask = self.geometry_augmenter(images, masks)

        masks = [MultiInstanceMask(m) for m in masks]

        new_width, new_height, _ = compute_resize_params(images[0], cfg.INPUT.MIN_DIM, cfg.INPUT.MAX_DIM)

        # convert images to torch Tensors
        images = torch.stack(self.np_to_tensor(*images), 0).float()

        # resize and pad images images
        images = F.interpolate(images, (new_height, new_width), mode='bilinear', align_corners=False)

        # scale and normalize images
        images = scale_and_normalize_images(images, cfg.INPUT.IMAGE_MEAN, cfg.INPUT.IMAGE_STD,
                                            invert_channels=cfg.INPUT.RGB_INPUT,
                                            normalize_to_unit_scale=cfg.INPUT.NORMALIZE_TO_UNIT_SCALE)  # [T, C, H, W]

        invalid_pts_mask = torch.stack([torch.from_numpy(m) for m in invalid_pts_mask]).unsqueeze(1)  # [T, 1, H, W]
        invalid_pts_mask = F.interpolate(invalid_pts_mask.float(), (new_height, new_width), mode='bilinear', align_corners=False) > 0.5
        images = torch.where(invalid_pts_mask, torch.zeros_like(images), images)

        # resize masks
        masks = [m.resize((new_height, new_width)).tensor().long() for m in masks]  # List(array[H, W])

        meta_info = {
            'height': image_height,
            'width': image_width
        }

        return {
            "images": images,  # [T, C, H, W]
            "masks": masks,  # [T+1, H, W]
            "instance_presence_map": torch.from_numpy(instance_presence_map),  # [T, I]
            "meta": meta_info
        }

    def get_mask_presence_map(self, masks: List[np.ndarray], num_instances: int):
        if cfg.TRAINING.APPLY_LOSS_ON_INTERMEDIATE_FRAMES:
            masks = np.stack(masks, 0).reshape(len(masks), -1)[:, None, :]  # [T, 1, H*W]
            instance_ids = np.arange(1, num_instances + 1)[None, :, None]  # [1, I, 1]
            return np.any(masks == instance_ids, 2)  # [T, I]
        else:
            return np.ones((len(masks), num_instances), bool)

    def apply_random_flip(self, images: List[np.ndarray], masks: List[np.ndarray]):
        if random.random() < 0.5:
            images = [np.flip(image, axis=1) for image in images]
            masks = [np.flip(m, axis=1) for m in masks]

        return images, masks

    def create_training_subsequences(self, sequences: List[VideoSequence], num_subsequences: int,
                                     num_instances: int):

        # fix seed so that same set of samples is generated across all processes
        rnd_state_backup = random.getstate()
        random.seed(2202)

        samples_by_num_instance = defaultdict(list)

        for seq in sequences:
            last_t = len(seq) - self.num_other_frames - self.min_delta_t

            for t in range(last_t):
                # if cfg.TRAINING.GT_SKIP_FRAMES > 0 and t % cfg.TRAINING.GT_SKIP_FRAMES != 0:
                #     continue

                valid_instance_ids = [iid for iid in seq.instance_ids if iid in seq.tracks[t]]

                if not valid_instance_ids:
                    continue

                bin_id = min(num_instances, len(valid_instance_ids))
                samples_by_num_instance[bin_id].append((seq.id, t, valid_instance_ids))

        # print("Pool size: ", sum([len(l) for l in samples_by_num_instance.values()]))

        train_samples = []
        train_sample_dims = []
        train_sample_ni = []

        sequences = {seq.id: seq for seq in sequences}
        num_instances_per_count = int(math.ceil(num_subsequences / float(num_instances)))
        available_sample_pool = []

        for ni in range(num_instances, 0, -1):
            available_sample_pool = samples_by_num_instance[ni] + available_sample_pool

            for ii in range(num_instances_per_count):
                ii = ii % len(available_sample_pool)

                seq_id, ref_frame_idx, instance_ids = available_sample_pool[ii]

                assert len(instance_ids) >= ni

                if len(instance_ids) > ni:
                    instance_ids = random.sample(instance_ids, ni)

                # sample other frames
                seq = sequences[seq_id]
                other_frames_list = list(
                    range(ref_frame_idx + self.min_delta_t, min(len(seq), ref_frame_idx + self.max_delta_t))
                )

                assert len(other_frames_list) >= self.num_other_frames, \
                    f"Something went wrong here: {ref_frame_idx}, {len(seq)}, {other_frames_list}"

                other_frame_idxes = sorted(random.sample(other_frames_list, self.num_other_frames))

                train_samples.append(self.compress_sample({
                    "seq_id": seq_id,
                    "ref_frame": ref_frame_idx,
                    "other_frames": other_frame_idxes,
                    "ref_instance_ids": instance_ids
                }))

                train_sample_dims.append((seq.height, seq.width))
                train_sample_ni.append(ni)

                if len(train_samples) % 200000 == 0:
                    print(f"Generated {len(train_samples)}/{num_subsequences} samples...")

        # restore initial random state
        random.setstate(rnd_state_backup)

        train_samples = train_samples[:num_subsequences]
        train_sample_dims = train_sample_dims[:num_subsequences]
        train_sample_ni = train_sample_ni[:num_subsequences]

        return train_samples, train_sample_dims, train_sample_ni

    # def create_training_subsequences_one_per_seq(self, sequences: List[VideoSequence], num_subsequences: int,
    #                                              num_instances: int):
    #
    #     # fix seed so that same set of samples is generated across all processes
    #     rnd_state_backup = random.getstate()
    #     random.seed(2202)
    #
    #     # samples_by_num_instance = defaultdict(list)
    #     annotated_samples = []
    #     # sequences = [seq for seq in sequences if len(seq) > (2 * self.num_other_frames)]
    #
    #     for seq in sequences:
    #         areas = seq.compute_instance_mask_areas()
    #         bboxes = seq.load_boxes(null_box=[0, 0, 0, 0])
    #
    #         # choose a good single frame from the video
    #         valid_instance_ids = defaultdict(list)
    #
    #         #for t in range(self.num_other_frames, len(seq) - self.num_other_frames):
    #         for t in range(0, len(seq)):
    #
    #             for i, iid in enumerate(seq.instance_ids):
    #                 if areas[t][i] < self.min_ref_mask_area:
    #                     continue
    #
    #                 if min(bboxes[t][i][2:]) < self.min_ref_mask_dim:
    #                     continue
    #
    #                 valid_instance_ids[t].append(iid)
    #
    #         if len(valid_instance_ids) == 0:
    #             continue
    #
    #         # take subset of frames with highest number of instances
    #         max_objects = max([len(iids) for iids in valid_instance_ids.values()])
    #         feasible_frame_set = [t for t in valid_instance_ids if len(valid_instance_ids[t]) == max_objects]
    #
    #         # choose frame closest to center
    #         delta_to_center = [abs(t - (len(seq) / 2.0)) for t in feasible_frame_set]
    #         chosen_frame_id = feasible_frame_set[torch.as_tensor(delta_to_center).argmin().item()]
    #         annotated_samples.append((seq.id, chosen_frame_id, valid_instance_ids[chosen_frame_id]))
    #
    #     train_samples = []
    #     train_sample_dims = []
    #     train_sample_ni = []
    #
    #     print(f"Base sample set: {len(annotated_samples)}")
    #
    #     sequences = {seq.id: seq for seq in sequences}
    #     num_samples_per_seq = int(math.ceil(num_subsequences / float(len(annotated_samples))))
    #
    #     if "davis" in self.dataset_name:
    #         # MAX_FRAME_GAP = 15
    #         MAX_FRAME_GAP = 6
    #     else:
    #         # MAX_FRAME_GAP = 6
    #         MAX_FRAME_GAP = 3
    #
    #     for seq_id, ref_frame, instance_ids in annotated_samples:
    #         seq = sequences[seq_id]
    #
    #         if ref_frame == 0:
    #             other_frame_set_earlier = []
    #         else:
    #             other_frame_set_earlier = torch.arange(ref_frame - 1, max(-1, ref_frame - MAX_FRAME_GAP - 1), -1).tolist()
    #
    #         if ref_frame == len(seq) - 1:
    #             other_frame_set_later = []
    #         else:
    #             other_frame_set_later = torch.arange(ref_frame + 1, min(ref_frame + MAX_FRAME_GAP + 1, len(seq))).tolist()
    #         # num_objects = len(instance_ids)
    #
    #         for _ in range(num_samples_per_seq):
    #             try:
    #                 if len(other_frame_set_later) < self.num_other_frames:
    #                     assert len(other_frame_set_earlier) >= self.num_other_frames
    #                     chosen_set = other_frame_set_earlier
    #
    #                 elif len(other_frame_set_earlier) < self.num_other_frames:
    #                     assert len(other_frame_set_later) >= self.num_other_frames
    #                     chosen_set = other_frame_set_later
    #
    #                 else:
    #                     chosen_set = other_frame_set_earlier if random.random() < 0.5 else other_frame_set_later
    #
    #             except AssertionError as err:
    #                 print(f"seq_len={len(seq)}, ref_frame={ref_frame}, other_frame_set_earlier={other_frame_set_earlier}, other_frame_set_later={other_frame_set_later}")
    #                 raise err
    #
    #             other_frames = random.sample(chosen_set, self.num_other_frames)
    #
    #             if len(instance_ids) > num_instances:
    #                 chosen_instance_ids = random.sample(instance_ids, num_instances)
    #             else:
    #                 chosen_instance_ids = instance_ids
    #
    #             train_samples.append(self.compress_sample({
    #                 "seq_id": seq_id,
    #                 "ref_frame": ref_frame,
    #                 "other_frames": other_frames,
    #                 "ref_instance_ids": chosen_instance_ids
    #             }))
    #
    #             train_sample_dims.append((seq.height, seq.width))
    #             train_sample_ni.append(len(chosen_instance_ids))
    #
    #             if len(train_samples) % 200000 == 0:
    #                 print(f"Generated {len(train_samples)}/{num_subsequences} samples...")
    #
    #     # --- debug ----
    #     # print(self.dataset_name)
    #     # print([(ni, (torch.as_tensor(train_sample_ni) == ni).sum().item()) for ni in list(range(1, num_instances + 1))])
    #     # --------------
    #
    #     # restore initial random state
    #     random.setstate(rnd_state_backup)
    #
    #     train_samples = train_samples[:num_subsequences]
    #     train_sample_dims = train_sample_dims[:num_subsequences]
    #     train_sample_ni = train_sample_ni[:num_subsequences]
    #
    #     return train_samples, train_sample_dims, train_sample_ni
    #
    # # This is only for a weird experiment. Do not use for regular training!
    # def create_training_subsequences_from_sparse(self, sequences: List[GenericVideoSequence], num_subsequences: int,
    #                                              num_instances: int, sparseness_factor: int):
    #
    #     # fix seed so that same set of samples is generated across all processes
    #     rnd_state_backup = random.getstate()
    #     random.seed(2202)
    #     # assert sparseness_factor == 10
    #
    #     samples_by_num_instance = defaultdict(list)
    #     seq_len = self.num_other_frames + 1
    #
    #     sequences = [seq for seq in sequences if len(seq) >= (seq_len * sparseness_factor) + 1]
    #
    #     for seq in sequences:
    #         areas = seq.compute_instance_mask_areas()
    #         bboxes = seq.load_boxes(null_box=[0, 0, 0, 0])
    #
    #         last_t = len(seq) - self.num_other_frames - self.min_delta_t
    #
    #         for t in range(last_t):
    #             if t % sparseness_factor != 0:
    #                 continue
    #
    #             # if t < (10 * self.num_other_frames) and t + (10 * self.num_other_frames) >= len(seq):
    #             #     continue
    #
    #             valid_instance_ids = []
    #
    #             for i, iid in enumerate(seq.instance_ids):
    #                 if areas[t][i] < self.min_ref_mask_area:
    #                     continue
    #
    #                 if min(bboxes[t][i][2:]) < self.min_ref_mask_dim:
    #                     continue
    #
    #                 valid_instance_ids.append(iid)
    #
    #             if not valid_instance_ids:
    #                 continue
    #
    #             bin_id = min(num_instances, len(valid_instance_ids))
    #             samples_by_num_instance[bin_id].append((seq.id, t, valid_instance_ids))
    #
    #     # print("Pool size: ", sum([len(l) for l in samples_by_num_instance.values()]))
    #
    #     train_samples = []
    #     train_sample_dims = []
    #     train_sample_ni = []
    #
    #     sequences = {seq.id: seq for seq in sequences}
    #     num_instances_per_count = int(math.ceil(num_subsequences / float(num_instances)))
    #     available_sample_pool = []
    #
    #     for ni in range(num_instances, 0, -1):
    #         available_sample_pool = samples_by_num_instance[ni] + available_sample_pool
    #
    #         for ii in range(num_instances_per_count):
    #             ii = ii % len(available_sample_pool)
    #
    #             seq_id, ref_frame_idx, instance_ids = available_sample_pool[ii]
    #
    #             assert len(instance_ids) >= ni
    #
    #             if len(instance_ids) > ni:
    #                 instance_ids = random.sample(instance_ids, ni)
    #
    #             # sample other frames
    #             seq = sequences[seq_id]
    #
    #             if ref_frame_idx < (10 * self.num_other_frames):
    #                 # can only sample ahead
    #                 other_frame_idxes = torch.arange(ref_frame_idx + 10, ref_frame_idx + (10 * seq_len), 10).tolist()
    #
    #             elif ref_frame_idx + (10 * self.num_other_frames) >= len(seq):
    #                 # can only sample behind
    #                 other_frame_idxes = torch.arange(ref_frame_idx - 10, ref_frame_idx - (10 * seq_len), -10).tolist()
    #
    #             else:
    #                 # randomly decide
    #                 if random.random() < 0.5:
    #                     other_frame_idxes = torch.arange(ref_frame_idx + 10, ref_frame_idx + (10 * seq_len), 10).tolist()
    #                 else:
    #                     other_frame_idxes = torch.arange(ref_frame_idx - 10, ref_frame_idx - (10 * seq_len), -10).tolist()
    #
    #             assert len(other_frame_idxes) == self.num_other_frames, \
    #                 f"{len(other_frame_idxes)}, {self.num_other_frames}"
    #
    #             assert all([0 <= t < len(seq) for t in other_frame_idxes]), \
    #                 f"ref_frame_idx={ref_frame_idx}, other_frame_idxes={other_frame_idxes}, sequence length={len(seq)}"
    #
    #             # assert len(other_frames_list) >= self.num_other_frames, \
    #             #     f"Something went wrong here: {ref_frame_idx}, {len(seq)}, {other_frames_list}"
    #
    #             # other_frame_idxes = sorted(random.sample(other_frames_list, self.num_other_frames))
    #
    #             train_samples.append(self.compress_sample({
    #                 "seq_id": seq_id,
    #                 "ref_frame": ref_frame_idx,
    #                 "other_frames": other_frame_idxes,
    #                 "ref_instance_ids": instance_ids
    #             }))
    #
    #             train_sample_dims.append((seq.height, seq.width))
    #             train_sample_ni.append(ni)
    #
    #             if len(train_samples) % 200000 == 0:
    #                 print(f"Generated {len(train_samples)}/{num_subsequences} samples...")
    #
    #     # restore initial random state
    #     random.setstate(rnd_state_backup)
    #
    #     train_samples = train_samples[:num_subsequences]
    #     train_sample_dims = train_sample_dims[:num_subsequences]
    #     train_sample_ni = train_sample_ni[:num_subsequences]
    #
    #     return train_samples, train_sample_dims, train_sample_ni
