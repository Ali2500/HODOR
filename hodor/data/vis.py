from hodor.config import cfg
from hodor.data.images import CocoDataset
from hodor.data.vos import DenseVideoDataset, SparseVideoDataset
from hodor.utils.paths import Paths
from hodor.data.utils.preprocessing import collate_fn_train, reverse_image_scaling_and_normalization

from argparse import ArgumentParser
from torch.utils.data import DataLoader

import cv2
import numpy as np
import os.path as osp
import torch
import random


def _overlay_mask(image, mask, mask_opacity=0.6, mask_color=(0, 255, 0)):
    if mask.ndim == 3:
        assert mask.shape[2] == 1
        _mask = mask.squeeze(axis=2)
    else:
        _mask = mask
    mask_bgr = np.stack((_mask, _mask, _mask), axis=2)
    masked_image = np.where(mask_bgr > 0, mask_color, image)
    return ((mask_opacity * masked_image) + ((1. - mask_opacity) * image)).astype(np.uint8)


def overlay_instance_masks_on_image(image, instance_mask, instance_ids, cmap):
    for ii in instance_ids:
        image = _overlay_mask(image, instance_mask == ii, mask_color=tuple(cmap[ii+1]), mask_opacity=0.5)
    return image


def create_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def main(args):
    if args.dataset == "coco":
        dataset = CocoDataset(Paths.coco_train_images(), Paths.coco_train_anns(), 1000,
                              max_num_instances=cfg.TRAINING.NUM_INSTANCES,
                              resize_mode="px")

    elif args.dataset == "ytvos_gt":
        dataset = DenseVideoDataset(Paths.youtube_vos_train_images_dir(), Paths.youtube_vos_train_anns(), "ytvos", 1000,
                                    num_instances=cfg.TRAINING.NUM_INSTANCES,
                                    min_delta_t=cfg.TRAINING.MIN_CLIP_DELTA,
                                    max_delta_t=cfg.TRAINING.MAX_CLIP_DELTA,
                                    min_ref_mask_area=cfg.TRAINING.MIN_REF_MASK_AREA,
                                    min_ref_mask_dim=cfg.TRAINING.MIN_REF_MASK_DIM,
                                    apply_color_augmentations=cfg.TRAINING.AUGMENTATIONS.COLOR
                                    )

    elif args.dataset == "davis_gt":
        dataset = DenseVideoDataset(Paths.davis_base_dir(), Paths.davis_train_anns(), "davis", 1000,
                                    num_instances=cfg.TRAINING.NUM_INSTANCES,
                                    min_delta_t=cfg.TRAINING.MIN_CLIP_DELTA,
                                    max_delta_t=cfg.TRAINING.MAX_CLIP_DELTA,
                                    min_ref_mask_area=cfg.TRAINING.MIN_REF_MASK_AREA,
                                    min_ref_mask_dim=cfg.TRAINING.MIN_REF_MASK_DIM,
                                    apply_color_augmentations=cfg.TRAINING.AUGMENTATIONS.COLOR
                                    )

    elif args.dataset.endswith("_sparse"):
        if args.dataset == "davis_sparse":
            base_dir = Paths.davis_base_dir()
            anns_path = Paths.davis_train_anns()
        elif args.dataset == "ytvos_sparse":
            base_dir = Paths.youtube_vos_train_images_dir()
            anns_path = Paths.youtube_vos_train_anns()
        else:
            raise ValueError("Should not be here")

        dataset = SparseVideoDataset(base_dir, anns_path, args.dataset.split("_")[0], 1000,
                                     num_instances=cfg.TRAINING.NUM_INSTANCES,
                                     min_delta_t=cfg.TRAINING.MIN_CLIP_DELTA,
                                     max_delta_t=cfg.TRAINING.MAX_CLIP_DELTA,
                                     min_ref_mask_area=cfg.TRAINING.MIN_REF_MASK_AREA,
                                     min_ref_mask_dim=cfg.TRAINING.MIN_REF_MASK_DIM,
                                     apply_color_augmentations=cfg.TRAINING.AUGMENTATIONS.COLOR
                                     )

    else:
        raise ValueError(f"Invalid dataset {args.dataset}")

    data_loader = DataLoader(dataset, 1, shuffle=args.shuffle, collate_fn=collate_fn_train, num_workers=args.num_workers)

    cmap = create_color_map().tolist()

    cv2.namedWindow('Ref Image', cv2.WINDOW_NORMAL)
    for t in range(cfg.TRAINING.NUM_OTHER_FRAMES):
        cv2.namedWindow(f'Other {t+1}', cv2.WINDOW_NORMAL)

    for sample in data_loader:
        ref_image = reverse_image_scaling_and_normalization(
            sample['ref_image'], cfg.INPUT.IMAGE_MEAN, cfg.INPUT.IMAGE_STD, invert_channels=cfg.INPUT.RGB_INPUT,
            normalize_to_unit_scale=cfg.INPUT.NORMALIZE_TO_UNIT_SCALE)[0]  # [C, H, W]

        ref_image = ref_image.permute(1, 2, 0).round().byte().numpy()  # [H, W]

        ref_patch_mask = sample['ref_mask'][0].numpy()
        instance_ids = list(set(np.unique(ref_patch_mask).tolist()) - {0})

        if sample['other_images'] is not None:
            other_images = reverse_image_scaling_and_normalization(
                sample['other_images'][0], cfg.INPUT.IMAGE_MEAN, cfg.INPUT.IMAGE_STD, invert_channels=cfg.INPUT.RGB_INPUT,
                normalize_to_unit_scale=cfg.INPUT.NORMALIZE_TO_UNIT_SCALE)  # [T, C, H, W]

            other_images = other_images.permute(0, 2, 3, 1)
            other_images = other_images.round().byte().numpy()

        other_masks = None
        if torch.is_tensor(sample['other_masks']):
            other_masks = sample['other_masks'][0].byte().numpy()  # [T, H, W]

        assert ref_image.shape[:2] == ref_patch_mask.shape, f"Shape mismatch: {ref_image.shape} vs. {ref_patch_mask.shape}"

        ref_image = overlay_instance_masks_on_image(ref_image, ref_patch_mask, instance_ids, cmap)
        cv2.imshow('Ref Image', ref_image)

        for t in range(cfg.TRAINING.NUM_OTHER_FRAMES):
            if other_masks is None:
                q_im = other_images[t]
            else:
                q_im = overlay_instance_masks_on_image(other_images[t], other_masks[t], instance_ids, cmap)

            cv2.imshow(f'Other {t+1}', q_im)

        print("META: {}".format(sample["meta"][0]))
        if cv2.waitKey(0) == 113:
            exit(0)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dataset', '-d', required=True)
    parser.add_argument('--cfg', required=False)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)

    random.seed(42)
    torch.manual_seed(42)

    parser_args = parser.parse_args()
    if parser_args.cfg:
        cfg.merge_from_file(osp.join(Paths.configs_dir(), parser_args.cfg))

    main(parser_args)
