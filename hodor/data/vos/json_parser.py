from typing import List, Tuple, Dict, Any, Union
from pycocotools import mask as masktools

import cv2
import json
import numpy as np
import os


class ObjectAnnotation(object):
    def __init__(self, obj_dict: Dict[str, Any]):
        self.bbox: List[int] = obj_dict['box']
        self.rle: str = obj_dict['seg']
        self.area: int = int(obj_dict['area'])

    def load_mask(self, height, width) -> np.ndarray:
        return np.ascontiguousarray(masktools.decode({
            "counts": self.rle.encode("utf-8"),
            "size": (height, width)
        })).astype(np.uint8)


class VideoSequence(object):
    def __init__(self, seq_dict, base_dir):
        self.base_dir = base_dir
        self.image_paths = seq_dict["image_paths"]
        self.image_dims = (int(seq_dict["height"]), int(seq_dict["width"]))
        self.id = seq_dict["id"]
        self.instance_categories = {int(track_id): cat_id for track_id, cat_id in seq_dict["categories"].items()}

        self.tracks = []
        for t in range(len(seq_dict["tracks"])):
            self.tracks.append({
                int(inst_id): ObjectAnnotation(obj_ann_dict) for inst_id, obj_ann_dict in seq_dict["tracks"][t].items()
            })

    @property
    def height(self):
        return self.image_dims[0]

    @property
    def width(self):
        return self.image_dims[1]

    @property
    def instance_ids(self):
        return list(self.instance_categories.keys())

    @property
    def category_labels(self):
        return [self.instance_categories[instance_id] for instance_id in self.instance_ids]

    @property
    def num_obj_annotations(self):
        return sum([len(tracks_t) for tracks_t in self.tracks])

    def __len__(self):
        return len(self.image_paths)

    def load_images(self, frame_idxes=None):
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        images = []
        for t in frame_idxes:
            im = cv2.imread(os.path.join(self.base_dir, self.image_paths[t]), cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError("No image found at path: {}".format(os.path.join(self.base_dir, self.image_paths[t])))
            images.append(im)

        return images

    def load_masks(self, frame_idxes=None, instance_ids=None, absent_instance_behavior="return_none") -> \
            List[List[Union[np.ndarray, None]]]:
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        if instance_ids is None:
            instance_ids = self.instance_ids

        masks = []
        if absent_instance_behavior in ("return_none", "raise_error"):
            null_mask = None
        elif absent_instance_behavior == "return_zeros_mask":
            null_mask = np.zeros((self.height, self.width), np.uint8)
        else:
            raise ValueError(f"Invalid argument '{absent_instance_behavior}'")

        for t in frame_idxes:
            masks_t = []

            for instance_id in instance_ids:
                if instance_id in self.tracks[t]:
                    masks_t.append(self.tracks[t][instance_id].load_mask(self.height, self.width))
                else:
                    if absent_instance_behavior == "raise_error":
                        raise ValueError(f"Instance ID {instance_id} not found at frame {t} in sequence {self.id}")
                    masks_t.append(null_mask)

            masks.append(masks_t)

        return masks

    def filter_objects_by_size(self, min_ref_mask_area: int, min_ref_mask_dim: int):
        for t in range(len(self)):
            self.tracks[t] = {
                inst_id: obj for inst_id, obj in self.tracks[t].items()
                if obj.area >= min_ref_mask_area and min(obj.bbox[2:]) >= min_ref_mask_dim
            }


def parse_generic_video_dataset(base_dir, dataset_json) -> Tuple[List[VideoSequence], Dict[str, Any]]:
    with open(dataset_json, 'r') as fh:
        dataset = json.load(fh)

    seqs = [VideoSequence(seq, base_dir) for seq in dataset["sequences"]]
    meta_info = dataset["meta"]
    meta_info["category_labels"] = {int(k): v for k, v in meta_info["category_labels"].items()}

    return seqs, meta_info
