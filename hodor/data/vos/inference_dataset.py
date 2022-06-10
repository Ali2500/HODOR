from hodor.config import cfg
from hodor.data.utils.preprocessing import compute_resize_params, scale_and_normalize_images, ToTorchTensor, \
    BatchImageTransform

import cv2
import pycocotools.mask as mt
import numpy as np
import os.path as osp
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class VOSInferenceDataset(Dataset):
    def __init__(self, base_dir, vds_json_file, name, seq_ids=None):
        super().__init__()

        self.base_dir = base_dir
        self.vds_json_file = vds_json_file
        self.name = name

        with open(vds_json_file, 'r') as fh:
            sequences = json.load(fh)['sequences']

        if seq_ids is not None:
            self.sequences = [s for s in sequences if s['id'] in seq_ids]
            assert len(self.sequences) > 0, "Zero sequences remaining after filtering"
        else:
            self.sequences = sequences

        self.eval_files = dict()
        for seq in self.sequences:
            self.eval_files[seq['id']] = seq.get("eval_files", None)

        self.np_to_tensor = BatchImageTransform(
            ToTorchTensor(format='CHW')
        )

    @staticmethod
    def _read_image(im_path: str) -> np.ndarray:
        image = cv2.imread(im_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"No image exists at expected path: {im_path}")
        return image

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]

        # load images
        images = [self._read_image(osp.join(self.base_dir, rel_path)) for rel_path in seq['image_paths']]

        img_h, img_w = images[0].shape[:2]

        # load masks
        ref_mask_frame_ids = []
        instance_ids = []
        ref_masks = []

        for frame_id, segs in seq['segmentations'].items():
            for instance_id, seg in segs.items():
                ref_mask_frame_ids.append(int(frame_id))
                instance_ids.append(int(instance_id))

                ref_masks.append(torch.from_numpy(mt.decode({
                    "counts": seg.encode("utf-8"),
                    "size": (img_h, img_w)
                }).astype(np.uint8)))

        ref_masks = torch.stack(ref_masks, 0)  # [N, H, W]

        # compute scale factor for mask resizing
        new_width, new_height, _ = compute_resize_params(images[0], cfg.INPUT.MIN_DIM, cfg.INPUT.MAX_DIM)

        # convert images to torch Tensors
        images = torch.stack(self.np_to_tensor(*images), 0).float()

        # resize images and masks
        images = F.interpolate(images, (new_height, new_width), mode='bilinear', align_corners=False)
        ref_masks = F.interpolate(ref_masks[None].float(), (new_height, new_width), mode='bilinear', align_corners=False)
        ref_masks = (ref_masks.squeeze(0) > 0.5).byte()

        # scale and normalize images
        images = scale_and_normalize_images(images, cfg.INPUT.IMAGE_MEAN, cfg.INPUT.IMAGE_STD,
                                            invert_channels=cfg.INPUT.RGB_INPUT,
                                            normalize_to_unit_scale=cfg.INPUT.NORMALIZE_TO_UNIT_SCALE)

        filenames = [osp.splitext(osp.split(rel_path)[-1])[0] for rel_path in seq['image_paths']]

        meta_info = {
            'seq_name': seq['id'],
            'height': img_h,
            'width': img_w,
            'resized_height': new_height,
            'resized_width': new_width,
        }

        return {
            "images": images,
            'filenames': filenames,
            "ref_masks": ref_masks,
            'ref_mask_frame_nums': torch.as_tensor(ref_mask_frame_ids, dtype=torch.long),
            "instance_ids": torch.as_tensor(instance_ids, dtype=torch.long),
            "meta": meta_info
        }
