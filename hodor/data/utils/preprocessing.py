from typing import List, Tuple
from hodor.config import cfg

import math
import numpy as np
import torch
import torch.nn.functional as F


class ToTorchTensor:
    def __init__(self, format='HWC', dtype=None):
        assert format in ['HWC', 'CHW']
        self.format = format
        self.dtype = dtype

    def __call__(self, image):
        tensor = torch.from_numpy(np.ascontiguousarray(image))

        if tensor.ndimension() == 3 and self.format == 'CHW':
            tensor = tensor.permute(2, 0, 1)

        if self.dtype is not None:
            return tensor.to(dtype=self.dtype)
        else:
            return tensor


class BatchImageTransform(object):
    def __init__(self, transform):
        self.__transform = transform

    def __call__(self, *images):
        return [self.__transform(image) for image in images]


def scale_and_normalize_images(images, means, scales, invert_channels, normalize_to_unit_scale):
    """
    Scales and normalizes images
    :param images: tensor(T, C, H, W)
    :param means: list(float)
    :param scales: list(float)
    :param invert_channels: bool
    :return: tensor(T, C, H, W)
    """
    means = torch.tensor(means, dtype=torch.float32)[None, :, None, None]  # [1, 3, 1, 1]
    scales = torch.tensor(scales, dtype=torch.float32)[None, :, None, None]  # [1. 3. 1. 1]
    if normalize_to_unit_scale:
        images = images / 255.

    images = (images - means) / scales
    if invert_channels:
        return images.flip(dims=[1])
    else:
        return images


def reverse_image_scaling_and_normalization(images, means, scales, invert_channels, normalize_to_unit_scale):
    if invert_channels:
        images = images.flip(dims=[1])

    means = torch.tensor(means, dtype=torch.float32, device=images.device)[None, :, None, None]  # [1, 3, 1, 1]
    scales = torch.tensor(scales, dtype=torch.float32, device=images.device)[None, :, None, None]  # [1. 3. 1. 1]

    images = (images * scales) + means

    if normalize_to_unit_scale:
        images = images * 255.

    return images


def compute_resize_params(image, min_dim, max_dim):
    lower_size = float(min(image.shape[:2]))
    higher_size = float(max(image.shape[:2]))

    scale_factor = min_dim / lower_size
    if (higher_size * scale_factor) > max_dim:
        scale_factor = max_dim / higher_size

    height, width = image.shape[:2]
    new_height, new_width = round(scale_factor * height), round(scale_factor * width)

    return new_width, new_height, scale_factor


def compute_resize_params_from_pixel_counts(image_dims: Tuple[int, int], max_pixels: int) -> Tuple[int, int]:
    h, w = image_dims
    ar = float(h) / float(w)

    new_h = int(math.floor(math.sqrt(ar * max_pixels)))
    new_w = max_pixels // new_h

    return new_w, new_h


def compute_padded_dims(width, height):
    padded_width = (int(math.ceil(width / 32)) * 32)
    padded_height = (int(math.ceil(height / 32)) * 32)
    return padded_width, padded_height


def pad_tensor_list(x: List[torch.Tensor], padded_width: int, padded_height: int) -> torch.Tensor:
    for ii in range(len(x)):
        pad_right = padded_width - x[ii].shape[-1]
        pad_bottom = padded_height - x[ii].shape[-2]

        if x[ii].ndim == 2:
            x[ii] = F.pad(x[ii][None, None], (0, pad_right, 0, pad_bottom), mode='constant', value=0)[0, 0]
        elif x[ii].ndim == 3:
            x[ii] = F.pad(x[ii][None], (0, pad_right, 0, pad_bottom), mode='constant', value=0)[0]
        elif x[ii].ndim == 4:
            x[ii] = F.pad(x[ii], (0, pad_right, 0, pad_bottom), mode='constant', value=0)
        elif x[ii].ndim == 5:
            d1, d2 = x[ii].shape[:2]
            x[ii] = F.pad(x[ii].flatten(0, 1), (0, pad_right, 0, pad_bottom), mode='constant', value=0)
            x[ii] = x[ii].view(d1, d2, *x[ii].shape[1:])
        else:
            raise NotImplementedError("No implementation for ndims = {}".format(x[ii].ndim))

    return torch.stack(x, 0)


def collate_fn_train(samples):
    """Collate function for torch DataLoader

    Arguments:
        samples {list} -- List of training samples from a Dataset

    Returns:
        [type] -- [description]
    """
    ref_image = [s['ref_image'] for s in samples]
    ref_mask = [s['ref_mask'] for s in samples]
    meta = [s['meta'] for s in samples]
    other_images = [s['other_images'] for s in samples]
    # video_flags = torch.stack([s['video_flag'] for s in samples])
    instance_presence_map = torch.stack([s['instance_presence_map'] for s in samples], 0)  # [B, T, I]

    num_instances = [meta_s['num_instances'] for meta_s in meta]
    assert len(set(num_instances)) == 1, "All batch samples must have the same number of instances, but got {}".format(num_instances)

    heights, widths = zip(*[im.shape[-2:] for im in ref_image])

    padded_width, padded_height = compute_padded_dims(max(widths), max(heights))
    # print(f"{max(widths)}, {max(heights)} --> {padded_width}, {padded_height}")

    ref_image = pad_tensor_list(ref_image, padded_width, padded_height)
    ref_mask = pad_tensor_list(ref_mask, padded_width, padded_height)

    if cfg.TRAINING.NUM_OTHER_FRAMES == 0:
        other_images = None
    else:
        other_images = pad_tensor_list(other_images, padded_width, padded_height)

    if cfg.TRAINING.NUM_OTHER_FRAMES > 0 and cfg.TRAINING.INTERMEDIATE_FRAME_GT_AVAILABLE:
        other_masks = pad_tensor_list([s['other_masks'] for s in samples], padded_width, padded_height)
    else:
        other_masks = None

    return {
        "ref_image": ref_image,  # [B, C, H, W]
        "ref_mask": ref_mask,  # [B, H, W]
        "meta": meta,  # List[Dict],
        "other_images": other_images,
        "other_masks": other_masks,  # [B, T, H, W]
        "num_instances": num_instances[0],
        "instance_presence_map": instance_presence_map
    }


def collate_fn_inference(samples):
    """Collate function for torch DataLoader

    Arguments:
        samples {list} -- List of training samples from a Dataset

    Returns:
        [type] -- [description]
    """
    assert len(samples) == 1
    
    for s in samples:
        padded_width, padded_height = compute_padded_dims(s['images'].shape[-1], s['images'].shape[-2])
        pad_w, pad_h = padded_width - s['images'].shape[-1], padded_height - s['images'].shape[-2]

        s['images'] = F.pad(s['images'], (0, pad_w, 0, pad_h), mode='constant', value=0)
        s['ref_masks'] = F.pad(s['ref_masks'][None], (0, pad_w, 0, pad_h), mode='constant', value=0)[0]

    return samples[0]
