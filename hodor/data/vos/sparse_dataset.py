from typing import List, Dict
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


class SparseVideoDataset(object):
    def __init__(self, base_dir: str, json_path: str, name: str, samples_to_create: int, num_instances: int,
                 min_delta_t: int, max_delta_t: int, min_ref_mask_area: int, min_ref_mask_dim: int,
                 apply_color_augmentations: bool):

        assert samples_to_create > 0

        self.base_dir = base_dir
        sequences, meta_info = parse_generic_video_dataset(base_dir, json_path)

        for seq in sequences:
            seq.filter_objects_by_size(min_ref_mask_area, min_ref_mask_dim)

        self.sequences = {seq.id: seq for seq in sequences if seq.num_obj_annotations > 0}

        self.num_other_frames = cfg.TRAINING.NUM_OTHER_FRAMES
        assert max_delta_t - min_delta_t + 1 >= self.num_other_frames

        self.min_delta_t = min_delta_t
        self.max_delta_t = max_delta_t

        self.min_ref_mask_area = min_ref_mask_area
        self.min_ref_mask_dim = min_ref_mask_dim
        self.dataset_name = name

        self.samples, self.sample_image_dims, self.sample_num_instances = self.create_training_subsequences(
            samples_to_create, num_instances
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
        # print(seq.image_paths)

        all_frame_idxes = [sample["ref_frame"]] + sample["other_frames"]
        images = seq.load_images(all_frame_idxes)

        object_ids = sample["ref_instance_ids"]
        masks = seq.load_masks([sample["ref_frame"]], object_ids, absent_instance_behavior="raise_error")[0]

        ref_mask_combined = np.zeros((seq.height, seq.width), np.int8)

        for i, mask in enumerate(masks, 1):
            ref_mask_combined = np.where(mask, i, ref_mask_combined)

        parsed_dict = self.process_clip(images, ref_mask_combined, len(object_ids))

        ref_image = parsed_dict["images"][0]  # [C, H, W]
        other_images = parsed_dict["images"][1:]  # [T, C, H, W]

        ref_mask = parsed_dict["ref_mask"]  # [H, W]

        meta = parsed_dict["meta"]
        meta["dataset"] = self.dataset_name
        meta["seq_id"] = seq.id
        meta["frame_idxes"] = all_frame_idxes
        meta["num_instances"] = len(object_ids)

        return {
            "ref_image": ref_image,
            "ref_mask": ref_mask,
            "other_masks": None,
            "other_images": other_images,
            "instance_presence_map": parsed_dict["instance_presence_map"],
            "meta": meta
        }

    def process_clip(self, images: List[np.ndarray], mask: np.ndarray, num_instances: int):
        """Processes a video clip of images and masks

        Args:
            images (List[np.ndarray]): list of images, each of shape [H, W, C]
            mask (np.ndarray]: Reference frame mask of shape [H, W]
            num_instances: int

        Returns:
            Dict:
        """
        assert self.num_other_frames > 0

        image_height, image_width = images[0].shape[:2]

        # apply random horizontal flip and frame sequence reversal in training mode
        images, mask = self.apply_random_flip(images, mask)

        # apply color augmentation
        # augmenter = self.color_augmenter.to_deterministic()
        images = [self.color_augmenter.augment_image(image) for image in images]

        # apply geometric augmentations
        images, mask, invalid_pts_mask = self.geometry_augmenter(images, [mask] * len(images))  # hacky: copy masks to pass them through the API
        mask = MultiInstanceMask(mask[0])

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
        mask = mask.resize((new_height, new_width)).tensor().long()  # [H, W]

        meta_info = {
            'height': image_height,
            'width': image_width
        }

        return {
            "images": images,  # [T, C, H, W]
            "ref_mask": mask,  # [H, W]
            "instance_presence_map": torch.ones(images.size(0), num_instances, dtype=torch.bool),  # [T, I]
            "meta": meta_info
        }

    def apply_random_flip(self, images: List[np.ndarray], mask: np.ndarray):
        if random.random() < 0.5:
            images = [np.flip(image, axis=1) for image in images]
            mask = np.flip(mask, axis=1)

        return images, mask

    def create_training_subsequences(self, num_subsequences: int, num_instances: int):

        # fix seed so that same set of samples is generated across all processes
        rnd_state_backup = random.getstate()
        random.seed(2202)

        # samples_by_num_instance = defaultdict(list)
        feasible_sample_set = []
        # sequences = [seq for seq in sequences if len(seq) > (2 * self.num_other_frames)]

        for seq_id, seq in self.sequences.items():
            frame_ids_with_anns = [t for t in range(len(seq.tracks)) if len(seq.tracks[t]) > 0]

            if len(frame_ids_with_anns) > 1:
                # choose middle-most frame
                center_frame_id = int(round(float(len(seq))))
                distance_to_center = torch.as_tensor([abs(t - center_frame_id) for t in frame_ids_with_anns])
                min_idx = distance_to_center.argmin().item()
                chosen_frame_ids = [frame_ids_with_anns[min_idx]]

            else:
                chosen_frame_ids = frame_ids_with_anns

            for t in chosen_frame_ids:
                feasible_sample_set.append((seq_id, t, seq.tracks[t]))

        train_samples = []
        train_sample_dims = []
        train_sample_ni = []

        # print(f"Base sample set: {len(feasible_sample_set)}")

        # sequences = {seq.id: seq for seq in sequences}
        num_samples_per_seq = int(math.ceil(num_subsequences / float(len(feasible_sample_set))))

        for seq_id, ref_frame, frame_objects in feasible_sample_set:
            seq = self.sequences[seq_id]

            if ref_frame == 0:
                other_frame_set_earlier = []
            else:
                other_frame_set_earlier = torch.arange(ref_frame - 1, max(-1, ref_frame - self.max_delta_t - 1), -1).tolist()

            if ref_frame == len(seq) - 1:
                other_frame_set_later = []
            else:
                other_frame_set_later = torch.arange(ref_frame + 1, min(ref_frame + self.max_delta_t + 1, len(seq))).tolist()
            # num_objects = len(instance_ids)

            for _ in range(num_samples_per_seq):
                try:
                    if len(other_frame_set_later) < self.num_other_frames:
                        assert len(other_frame_set_earlier) >= self.num_other_frames
                        chosen_set = other_frame_set_earlier

                    elif len(other_frame_set_earlier) < self.num_other_frames:
                        assert len(other_frame_set_later) >= self.num_other_frames
                        chosen_set = other_frame_set_later

                    else:
                        chosen_set = other_frame_set_earlier if random.random() < 0.5 else other_frame_set_later

                except AssertionError as err:
                    print(f"seq_len={len(seq)}, ref_frame={ref_frame}, "
                          f"other_frame_set_earlier={other_frame_set_earlier}, "
                          f"other_frame_set_later={other_frame_set_later}")
                    raise err

                other_frames = random.sample(chosen_set, self.num_other_frames)

                assert all([0 <= t < len(seq) for t in other_frames]), \
                    f"Invalid other frames {other_frames} for sequence of length {len(seq)} (ref frame = {ref_frame})"

                object_ids = list(frame_objects.keys())

                if len(frame_objects) > num_instances:
                    chosen_instance_ids = random.sample(object_ids, num_instances)
                else:
                    chosen_instance_ids = object_ids

                train_samples.append(self.compress_sample({
                    "seq_id": seq_id,
                    "ref_frame": ref_frame,
                    "other_frames": other_frames,
                    "ref_instance_ids": chosen_instance_ids
                }))

                train_sample_dims.append((seq.height, seq.width))
                train_sample_ni.append(len(chosen_instance_ids))

                if len(train_samples) % 200000 == 0:
                    print(f"Generated {len(train_samples)}/{num_subsequences} samples...")

        # restore initial random state
        random.setstate(rnd_state_backup)

        train_samples = train_samples[:num_subsequences]
        train_sample_dims = train_sample_dims[:num_subsequences]
        train_sample_ni = train_sample_ni[:num_subsequences]

        return train_samples, train_sample_dims, train_sample_ni
