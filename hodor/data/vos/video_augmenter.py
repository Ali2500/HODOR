import cv2
import imgaug.augmenters as iaa
import numpy as np
import random
from typing import List, Tuple

from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from hodor.data.utils.mask_cropping import compute_mask_preserving_crop


class VideoAugmenter(object):
    def __init__(self, crop_factor: float, min_ref_mask_area: int):
        self.crop_factor = crop_factor
        self.min_ref_mask_area = min_ref_mask_area
        self.n_affine_tries = 3

        self.affine_augmenter = iaa.OneOf([
            iaa.Affine(rotate=(-20, 20), shear=(-10, 10)),
            iaa.PerspectiveTransform(scale=(0.08, 0.10))
        ])

    @staticmethod
    def get_mask_areas(mask: np.ndarray, num_instances: int):
        instance_ids = np.arange(1, num_instances + 1)[:, None]
        return np.sum((mask.reshape(-1)[None, :] == instance_ids).astype(np.float32), 1)

    @staticmethod
    def get_random_crop_params(img_h, img_w, crop_factor) -> Tuple[int, int, int, int]:
        assert 0. < crop_factor < 1.
        crop_h, crop_w = int(round(img_h * crop_factor)), int(round(img_w * crop_factor))

        start_x = random.randint(0, img_w - crop_w)
        start_y = random.randint(0, img_h - crop_h)

        return start_x, start_y, crop_w, crop_h

    @staticmethod
    def apply_crop(crop_params, args):
        x, y, w, h = crop_params

        if isinstance(args, (list, tuple)):
            cropped_args = [a[y:y + h, x:x + w] for a in args]
        elif isinstance(args, np.ndarray):
            cropped_args = args[y:y+h, x:x+w]
        else:
            raise RuntimeError(f"Invalid type: {type(args)}")

        return cropped_args

    def __call__(self, images: List[np.ndarray], masks: List[np.ndarray]):
        assert len(images) == len(masks)

        if random.random() < 0.5:
            invalid_pts_mask = [np.zeros(masks[0].shape, bool) for _ in range(len(images))]
            images, masks = self.augment_by_cropping(images, masks)
            return images, masks, invalid_pts_mask

        else:
            return self.augment_by_affine(images, masks)

    def augment_by_cropping(self, images: List[np.ndarray], masks: List[np.ndarray]) -> \
            Tuple[List[np.ndarray], List[np.ndarray]]:

        ref_mask = masks[0]
        img_h, img_w = ref_mask.shape

        # calculate instance mask areas
        num_instances = np.max(ref_mask)
        orig_instance_areas = self.get_mask_areas(ref_mask, num_instances)

        # (1) Attempt random crop
        for crop_factor in np.arange(self.crop_factor, 0.8, 0.1):
            crop_params = self.get_random_crop_params(img_h, img_w, crop_factor)
            cropped_ref_mask = self.apply_crop(crop_params, ref_mask)

            new_instance_areas = self.get_mask_areas(cropped_ref_mask, num_instances)
            area_ratios = new_instance_areas / orig_instance_areas

            if np.all(area_ratios >= 0.5) and np.all(new_instance_areas >= self.min_ref_mask_area):
                # sufficiently enough of every instance remains inside the cropped image
                # print("(1) worked")
                aug_masks = [cropped_ref_mask] + self.apply_crop(crop_params, masks[1:])
                aug_masks = [cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST_EXACT) for m in aug_masks]

                # resize to original dims
                aug_images = [
                    cv2.resize(im, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
                    for im in self.apply_crop(crop_params, images)
                ]

                return aug_images, aug_masks

        # (2) Too much of one or more instances was cropped out. Let's try to do a mask preserving crop
        for crop_factor in np.arange(self.crop_factor, 1.0, 0.1):
            target_dims = int(round(img_h * crop_factor)), int(round(img_w * crop_factor))
            output = compute_mask_preserving_crop(masks[0], target_dims)

            if output is not None:
                crop_params = output[0], output[1], target_dims[1], target_dims[0]

                aug_masks = [
                    cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST_EXACT)
                    for m in self.apply_crop(crop_params, masks)
                ]

                aug_images = [
                    cv2.resize(im, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
                    for im in self.apply_crop(crop_params, images)
                ]

                return aug_images, aug_masks

        # (3) Everything failed. return the original image and mask as is
        # print("Nothing worked :(")
        return images, masks

    def augment_by_affine(self, images: List[np.ndarray], masks: List[np.ndarray]) -> \
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

        ref_mask = masks[0]

        # calculate instance mask areas
        num_instances = np.max(ref_mask)
        orig_instance_areas = self.get_mask_areas(ref_mask, num_instances)

        accepted_augmenter = None
        ref_mask_seg = SegmentationMapsOnImage(ref_mask, ref_mask.shape)

        for _ in range(self.n_affine_tries):
            augmenter = self.affine_augmenter.to_deterministic()

            # augment ref mask
            aug_ref_mask = augmenter(segmentation_maps=ref_mask_seg).get_arr()

            new_instance_areas = self.get_mask_areas(aug_ref_mask, num_instances)
            area_ratios = new_instance_areas / orig_instance_areas

            if np.all(area_ratios >= 0.5) and np.all(new_instance_areas >= self.min_ref_mask_area):
                accepted_augmenter = augmenter
                break

        if accepted_augmenter is None:
            # no feasible augmentation found
            return images, masks, [np.zeros(ref_mask.shape, bool) for _ in range(len(images))]

        else:
            aug_image_mask_pairs = [
                accepted_augmenter(image=im, segmentation_maps=SegmentationMapsOnImage(m, ref_mask.shape))
                for im, m in zip(images, masks)
            ]

            aug_images = [pair[0] for pair in aug_image_mask_pairs]
            aug_masks = [pair[1].get_arr() for pair in aug_image_mask_pairs]

            invalid_pts_mask = accepted_augmenter(image=np.ones(list(ref_mask.shape) + [1], np.uint8)).squeeze(2) == 0
            invalid_pts_mask = [invalid_pts_mask for _ in range(len(images))]

            return aug_images, aug_masks, invalid_pts_mask
