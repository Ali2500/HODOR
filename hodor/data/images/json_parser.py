from pycocotools import mask as masktools
from typing import List, Tuple, Any, Dict

import cv2
import json
import os.path as osp
import numpy as np


class GenericImageSample(object):
    def __init__(self, base_dir, sample, img_id):
        self.id = sample.get('id', img_id)
        self.height = sample['height']
        self.width = sample['width']

        self.base_dir = base_dir
        self.path = sample['image_path']
        self.categories = [int(cat_id) for cat_id in sample['categories']]

        self.areas = sample.get("areas", None)
        self.bboxes = sample.get("bboxes", None)
        self.segmentations = sample.get('segmentations', None)
        self.ignore = sample.get("ignore", None)
        self.iscrowd = sample.get("iscrowd", None)

    @property
    def num_instances(self):
        return len(self.segmentations)

    def box_coordinates(self):
        if self.bboxes:
            return self.bboxes

        # get bounding box coordinates from RLE encoded masks
        assert self.segmentations
        rle_objs = [{
            "size": (self.height, self.width),
            "counts": seg.encode("utf-8")
        } for seg in self.segmentations]

        self.bboxes = masktools.toBbox(rle_objs)
        return self.bboxes

    def mask_areas(self):
        if self.areas is not None:
            return self.areas.copy()

        assert self.segmentations is not None
        rle_objs = [{
            "size": (self.height, self.width),
            "counts": seg.encode("utf-8")
        } for seg in self.segmentations]

        self.areas = [masktools.area([obj])[0] for obj in rle_objs]
        return self.areas.copy()

    def load_image(self):
        im = cv2.imread(osp.join(self.base_dir, self.path), cv2.IMREAD_COLOR)

        if im is None:
            raise ValueError("No image found at path: {}".format(self.path))
        return im

    def load_masks(self):
        return [np.ascontiguousarray(masktools.decode({
            "size": (self.height, self.width),
            "counts": seg.encode('utf-8')
        }).astype(np.uint8)) for seg in self.segmentations]


def parse_generic_image_dataset(base_dir, dataset_json) -> Tuple[List[GenericImageSample], Dict[str, Any]]:
    with open(dataset_json, 'r') as fh:
        dataset = json.load(fh)

    meta_info = dataset["meta"]

    # convert instance and category IDs from str to int
    if "category_labels" in meta_info:
        meta_info["category_labels"] = {
            int(k): v for k, v in meta_info["category_labels"].items()
        }

    samples = [GenericImageSample(base_dir, sample, ii) for ii, sample in enumerate(dataset["images"])]

    return samples, meta_info
