from typing import Tuple, Union, Optional
from torch import Tensor

import numpy as np
import torch
import random


@torch.no_grad()
def mask_to_bbox(masks: Tensor, raise_error_if_null_mask: Optional[bool] = True) -> torch.Tensor:
    """
    Extracts bounding boxes from masks
    :param masks: tensor of shape [N, H, W]
    :param raise_error_if_null_mask: Flag for whether or not to raise an error if a mask is all-zeros.
    :return: tensor of shape [N, 4] containing bounding boxes coordinates in [x, y, w, h] format.
             If `raise_error_if_null_mask` is False, coordinates [-1, -1, -1, -1] will be returned for all-zeros masks.
    """
    null_masks = torch.logical_not(torch.any(masks.flatten(1), 1))[:, None]  # [N, 1]
    if torch.any(null_masks) and raise_error_if_null_mask:
        raise ValueError("One or more all-zero masks found")

    h, w = masks.shape[-2:]

    reduced_rows = torch.any(masks, 2).long()  # [N, H]
    reduced_cols = torch.any(masks, 1).long()  # [N, W]

    x_min = (reduced_cols * torch.arange(-w-1, -1, dtype=torch.long, device=masks.device)[None]).argmin(1)  # [N]
    y_min = (reduced_rows * torch.arange(-h-1, -1, dtype=torch.long, device=masks.device)[None]).argmin(1)  # [N]

    x_max = (reduced_cols * torch.arange(w, dtype=torch.long, device=masks.device)[None]).argmax(1)  # [N]
    y_max = (reduced_rows * torch.arange(h, dtype=torch.long, device=masks.device)[None]).argmax(1)  # [N]

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    bbox_coords = torch.stack((x_min, y_min, width, height), 1)
    invalid_box = torch.full_like(bbox_coords, -1)

    return torch.where(null_masks, invalid_box, bbox_coords)


def _calculate_feasible_crop_range(mask_coord_1, mask_coord_2, im_len, min_crop_len):
    coord_1_set = list(range(0, min(mask_coord_1 + 1, im_len - min_crop_len + 1)))
    coord_2_set = list(range(max(mask_coord_2, min_crop_len), im_len + 1))

    if not coord_1_set or not coord_2_set:
        raise RuntimeError("No valid crop could be obtained")

    return random.choice(coord_1_set), random.choice(coord_2_set)


def compute_mask_preserving_crop(mask: Union[np.ndarray, Tensor], target_dims: Tuple[int, int]) -> Union[Tuple[int, int], None]:
    """
    Computes crop parameters for resizing the given `mask` to `target_dims` such that all nonzero points in `mask`
    remain preserved.
    :param mask: tensor of shape [H, W]
    :param target_dims: Desired size of crop as (height, width) tuple
    :return: Tuple of [x, y] values representing the top-left coordinate of the crop or None if no such crop is possible.
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(np.ascontiguousarray(mask))

    x1, y1, box_w, box_h = mask_to_bbox(mask.unsqueeze(0), raise_error_if_null_mask=True)[0].tolist()
    x2 = x1 + box_w
    y2 = y1 + box_h

    im_height, im_width = mask.shape
    crop_height, crop_width = target_dims

    if box_w >= crop_width or box_h >= crop_height:
        return None

    x1_min = max(0, x2 - crop_width)
    x1_max = min(im_width - crop_width, x1) + 1
    assert x1_max > x1_min, f"Invalid range for values of x1 for crop: [{x1_min}, {x1_max}]. Box dims: [{x1}, {y1}, {x2}, {y2}]. " \
        f"Image dims: [{im_height}, {im_width}]. Crop dims: [{crop_height}, {crop_width}]"

    y1_min = max(0, y2 - crop_height)
    y1_max = min(im_height - crop_height, y1) + 1
    assert y1_max > y1_min, f"Invalid range for values of y1 for crop: [{y1_min}, {y1_max}]. Box dims: [{x1}, {y1}, {x2}, {y2}]. " \
        f"Image dims: [{im_height}, {im_width}]. Crop dims: [{crop_height}, {crop_width}]"

    # sample x1 and y1 for crop
    crop_x1 = random.choice(list(range(x1_min, x1_max)))
    crop_y1 = random.choice(list(range(y1_min, y1_max)))

    return crop_x1, crop_y1
