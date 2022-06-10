import cv2
import imgaug.augmenters as iaa
import numpy as np
import random
from typing import List, Any

from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from hodor.data.utils.mask_cropping import compute_mask_preserving_crop


def get_augmenter(min_ref_mask_area: int):
    return ConcatAugmenter(
        [FancyAugmenter(0.6, min_ref_mask_area=min_ref_mask_area), BasicAugmenter()],
        [0.5, 0.5]
    )


class ConcatAugmenter:
    def __init__(self, augmenters: List[Any], probs: List[float]):
        assert sum(probs) == 1.0
        self.augmenters = augmenters
        self.cum_probs = np.cumsum(np.asarray(probs, np.float32))

    def __call__(self, image: np.ndarray, mask: np.ndarray, num_augmented_frames: int):
        val = random.random()
        for i, p in enumerate(self.cum_probs):
            if val <= p:
                return self.augmenters[i](image, mask, num_augmented_frames)

        raise ValueError("Should not be here")


class BasicAugmenter(object):
    def __init__(self, translate_range={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}, rotation_range=(-20, 20)):

        self.augmenter = iaa.OneOf([
            iaa.Affine(translate_percent=translate_range, rotate=rotation_range, shear=(-10, 10), backend="skimage"),
            iaa.PerspectiveTransform(scale=(0.10, 0.12)),
            iaa.Crop(percent=(0.1, 0.25), keep_size=True, sample_independently=False)
        ])

    def __call__(self, image: np.ndarray, mask: np.ndarray, num_frames: int):
        mask = SegmentationMapsOnImage(mask, shape=image.shape[:2])

        augmented_image_seq = []
        augmented_masks_seq = []
        invalid_pts_mask_seq = [np.zeros(image.shape[:2], bool)]
        # invalid_pts_mask_seq = []

        for _ in range(num_frames):
            det_augmenter = self.augmenter.to_deterministic()

            aug_image, aug_masks = det_augmenter(image=image, segmentation_maps=mask)
            invalid_pts_mask = det_augmenter(image=np.ones(image.shape[:2] + (1,), np.uint8)).squeeze(2)

            augmented_image_seq.append(aug_image)
            augmented_masks_seq.append(aug_masks.get_arr())
            invalid_pts_mask_seq.append(invalid_pts_mask == 0)

        return image, mask.get_arr(), augmented_image_seq, augmented_masks_seq, invalid_pts_mask_seq


class FancyAugmenter(object):
    def __init__(self, crop_factor: float, min_ref_mask_area: int):
        self.crop_factor = crop_factor
        self.min_ref_mask_area = min_ref_mask_area

    @staticmethod
    def get_mask_areas(mask: np.ndarray, num_instances: int):
        instance_ids = np.arange(1, num_instances + 1)[:, None]
        return np.sum((mask.reshape(-1)[None, :] == instance_ids).astype(np.float32), 1)

    @staticmethod
    def random_crop(image, mask, crop_factor):
        img_h, img_w = mask.shape
        crop_h, crop_w = int(round(img_h * crop_factor)), int(round(img_w * crop_factor))

        cropper = iaa.CropToFixedSize(width=crop_w, height=crop_h, position="uniform")
        image, mask = cropper(image=image, segmentation_maps=SegmentationMapsOnImage(mask, shape=mask.shape))

        return image, mask.get_arr()

    def augment_ref_frame(self, image: np.ndarray, mask: np.ndarray):
        img_h, img_w = mask.shape

        # calculate instance mask areas
        num_instances = np.max(mask)
        orig_instance_areas = self.get_mask_areas(mask, num_instances)

        # (1) Attempt random crop
        for crop_factor in np.arange(self.crop_factor, 0.8, 0.1):
            ref_image, ref_mask = self.random_crop(image, mask, crop_factor)

            new_instance_areas = self.get_mask_areas(ref_mask, num_instances)
            area_ratios = new_instance_areas / orig_instance_areas

            if np.all(area_ratios >= 0.5) and np.all(new_instance_areas >= self.min_ref_mask_area):
                # sufficiently enough of every instance remains inside the cropped image
                # print("(1) worked")
                # resize to original dims
                ref_image = cv2.resize(ref_image, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
                ref_mask = cv2.resize(ref_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST_EXACT)

                return ref_image, ref_mask

        # (2) Too much of one or more instances was cropped out. Let's try to do a mask preserving crop
        crop_params = None
        for crop_factor in np.arange(self.crop_factor, 1.0, 0.1):
            target_dims = int(round(img_h * crop_factor)), int(round(img_w * crop_factor))
            output = compute_mask_preserving_crop(mask, target_dims)

            if output is not None:
                crop_params = output[0], output[1], target_dims[1], target_dims[0]
                break

        if crop_params is not None:
            # mask preserving crop found
            x1, y1, w, h = crop_params

            ref_image = cv2.resize(image[y1:y1+h, x1:x1+w], (img_w, img_h), interpolation=cv2.INTER_CUBIC)
            ref_mask = cv2.resize(mask[y1:y1+h, x1:x1+w], (img_w, img_h), interpolation=cv2.INTER_NEAREST_EXACT)
            # print("(2) worked")
            return ref_image, ref_mask

        # (3) Everything failed. return the original image and mask as is
        # print("Nothing worked :(")
        return image, mask

    def __call__(self, image: np.ndarray, mask: np.ndarray, num_augmented_frames: int):
        ref_image, ref_mask = self.augment_ref_frame(image, mask)
        
        crop_factor = random.choice(np.arange(self.crop_factor, 0.8, 0.1).tolist())

        img_h, img_w = mask.shape
        crop_h, crop_w = int(round(img_h * crop_factor)), int(round(img_w * crop_factor))

        augmenter = iaa.Sequential([
            iaa.CropToFixedSize(width=crop_w, height=crop_h, position="uniform"),
            iaa.Resize({"height": img_h, "width": img_w}),
            iaa.OneOf([
                iaa.Affine(rotate=(-10, 10), shear=(-5, 5)),
                iaa.Identity()
            ])
        ])

        # augment other frames
        aug_images = []
        aug_masks = []
        invalid_pts_mask = [np.zeros((img_h, img_w), bool)]
        # invalid_pts_mask = []
        ones_array = np.ones((img_h, img_w, 1), np.uint8)

        for t in range(num_augmented_frames):
            augmenter_t = augmenter.to_deterministic()

            aug_image, aug_mask = augmenter_t(image=image, segmentation_maps=SegmentationMapsOnImage(mask, shape=mask.shape))
            aug_invalid_pts_mask = augmenter_t(image=ones_array).squeeze(2)

            aug_images.append(aug_image)
            aug_masks.append(aug_mask.get_arr())
            invalid_pts_mask.append(aug_invalid_pts_mask == 0)

        return ref_image, ref_mask, aug_images, aug_masks, invalid_pts_mask
