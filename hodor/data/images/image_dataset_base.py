from collections import defaultdict
from typing import List, Optional
from hodor.config import cfg
from hodor.data.utils.multi_instance_mask import MultiInstanceMask
from hodor.data.utils.preprocessing import compute_resize_params, scale_and_normalize_images, \
    compute_resize_params_from_pixel_counts, ToTorchTensor, BatchImageTransform
from hodor.data.images.image_augmenter import get_augmenter

import math
import numpy as np
import imgaug.augmenters as iaa
import random
import torch
import torch.nn.functional as F


class ImageDatasetBase(object):
    def __init__(self, name, min_ref_mask_area, min_ref_mask_dim, resize_mode):
        self.np_to_tensor = BatchImageTransform(
            ToTorchTensor(format='CHW')
        )

        self.ds_name = name
        self.min_ref_mask_area = min_ref_mask_area
        self.min_ref_mask_dim = min_ref_mask_dim

        self.augmenter = get_augmenter(min_ref_mask_area=min_ref_mask_area)

        self.num_augmented_frames = cfg.TRAINING.NUM_OTHER_FRAMES
        self.resize_mode = resize_mode

        self.base_color_augmenter = iaa.Sequential([
            iaa.AddToHueAndSaturation(value_hue=(-12, 12), value_saturation=(-12, 12)),
            iaa.LinearContrast(alpha=(0.95, 1.05)),
            iaa.AddToBrightness(add=(-25, 25))
        ])

        self.samples = []

    def create_training_samples(self, samples, num_samples, min_num_instances, num_instances):
        # fix seed so that same set of samples is generated across all processes
        rnd_state_baackup = random.getstate()
        random.seed(2202)

        # sort image samples into a dict based on number of instances.
        samples_by_num_instance = defaultdict(list)

        for s in samples:
            assert s.num_instances > 0
            if s.num_instances < min_num_instances:
                continue

            bin_id = min(num_instances, s.num_instances)
            samples_by_num_instance[bin_id].append(s)

        num_instances_per_count = int(math.ceil(num_samples / float(num_instances - min_num_instances + 1)))

        train_samples = []
        train_sample_dims = []
        train_sample_ni = []

        available_sample_pool = []

        for ni in range(num_instances, min_num_instances-1, -1):
            available_sample_pool = samples_by_num_instance[ni] + available_sample_pool

            for ii in range(num_instances_per_count):
                ii = ii % len(available_sample_pool)
                s = available_sample_pool[ii]

                assert s.num_instances >= ni
                if s.num_instances == ni:
                    instance_ids = list(range(s.num_instances))
                else:
                    instance_ids = random.sample(list(range(s.num_instances)), ni)

                train_samples.append({
                    "img_id": s.id,
                    "instance_ids": instance_ids
                })

                train_sample_dims.append((s.height, s.width))
                train_sample_ni.append(ni)

        # restore initial random state
        random.setstate(rnd_state_baackup)

        train_samples = train_samples[:num_samples]
        train_sample_dims = train_sample_dims[:num_samples]
        train_sample_ni = train_sample_ni[:num_samples]

        return train_samples, train_sample_dims, train_sample_ni

    def get_resized_dims(self, image):
        if self.resize_mode == "dims":
            w, h, _ = compute_resize_params(image, cfg.INPUT.MIN_DIM, cfg.INPUT.MAX_DIM)
            return w, h
        elif self.resize_mode == "px":
            return compute_resize_params_from_pixel_counts(image.shape[:2], cfg.INPUT.MAX_IMAGE_AREA)
        else:
            raise ValueError(f"Invalid resize mode: {self.resize_mode}")

    def process_image_mask_pair(self, image: np.ndarray, mask: np.ndarray):
        image = self.base_color_augmenter(image=image)
        num_instances = np.max(mask).item()

        image, mask, aug_images, aug_masks, invalid_pts_mask = self.augmenter(
            image, mask, self.num_augmented_frames
        )

        instance_presence_map = self.get_mask_presence_map([mask] + aug_masks, num_instances)

        if invalid_pts_mask:
            invalid_pts_mask = torch.from_numpy(np.stack(invalid_pts_mask, 0)).bool()  # [T, H, W]

        images = [image] + aug_images
        masks = [MultiInstanceMask(mask)] + [MultiInstanceMask(m) for m in aug_masks]

        image_height, image_width = image.shape[:2]

        # apply random horizontal flip and frame sequence reversal in training mode
        images, masks, invalid_pts_mask = self.apply_random_flip(images, masks, invalid_pts_mask)

        # compute scale factor for mask resizing
        # new_width, new_height, _ = compute_resize_params(image, cfg.INPUT.MIN_DIM, cfg.INPUT.MAX_DIM)
        new_width, new_height = self.get_resized_dims(image)

        # convert images to torch Tensors
        images = torch.stack(self.np_to_tensor(*images), 0).float()

        # resize and pad images images
        images = F.interpolate(images, (new_height, new_width), mode='bilinear', align_corners=False)

        # scale and normalize images
        images = scale_and_normalize_images(images, cfg.INPUT.IMAGE_MEAN, cfg.INPUT.IMAGE_STD,
                                            invert_channels=cfg.INPUT.RGB_INPUT,
                                            normalize_to_unit_scale=cfg.INPUT.NORMALIZE_TO_UNIT_SCALE)  # [T, C, H, W]

        # resize masks
        masks = torch.stack([m.resize((new_height, new_width)).tensor() for m in masks]).long()  # [T, H, W]

        if torch.is_tensor(invalid_pts_mask):
            invalid_pts_mask = F.interpolate(invalid_pts_mask.unsqueeze(1).float(), (new_height, new_width), mode='bilinear', align_corners=False) > 0.5  # [T, 1, H, W]
            images = torch.where(invalid_pts_mask, torch.zeros_like(images), images)

        ref_image = images[0]
        aug_images = images[1:] if images.size(0) > 1 else None

        ref_mask = masks[0]
        aug_masks = masks[1:] if masks.size(0) > 1 else None

        meta_info = {
            'height': image_height,
            'width': image_width,
            'dataset': self.ds_name,
        }

        return {
            "ref_image": ref_image,
            "other_images": aug_images,
            "ref_mask": ref_mask,
            "other_masks": aug_masks,
            "video_flag": torch.as_tensor(False, dtype=torch.bool),
            "instance_presence_map": torch.from_numpy(instance_presence_map),  # [T, I]
            "meta": meta_info
        }

    def apply_random_flip(self, images: List[np.ndarray], masks: List[MultiInstanceMask],
                          invalid_pts_mask: Optional[torch.Tensor] = None):
        if random.random() < 0.5:
            images = [np.flip(image, axis=1) for image in images]
            masks = [mask.flip_horizontal() for mask in masks]
            if torch.is_tensor(invalid_pts_mask):
                invalid_pts_mask = invalid_pts_mask.flip(dims=[2])

        return images, masks, invalid_pts_mask

    def get_mask_presence_map(self, masks: List[np.ndarray], num_instances: int):
        masks = np.stack(masks, 0).reshape(len(masks), -1)[:, None, :] # [T, 1, H*W]
        instance_ids = np.arange(1, num_instances + 1)[None, :, None]  # [1, I, 1]
        return np.any(masks == instance_ids, 2)  # [T, I]
