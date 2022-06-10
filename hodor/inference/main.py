from argparse import ArgumentParser
from collections import defaultdict
from hodor.config import cfg
from hodor.data.vos.inference_dataset import VOSInferenceDataset
from hodor.inference.inference_model import build_inference_model as build_model
from hodor.utils.paths import Paths
from hodor.data.utils.preprocessing import collate_fn_inference, reverse_image_scaling_and_normalization
from hodor.inference.timer import Timer
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from typing import List, Union

import cv2
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
import zipfile as zf


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, mask_opacity: float = 0.6, mask_color=(0, 255, 0)):
    if mask.ndim == 3:
        assert mask.shape[2] == 1
        _mask = mask.squeeze(axis=2)
    else:
        _mask = mask
    mask_bgr = np.stack((_mask, _mask, _mask), axis=2)
    masked_image = np.where(mask_bgr > 0, mask_color, image)
    return ((mask_opacity * masked_image) + ((1. - mask_opacity) * image)).astype(np.uint8)


def load_color_palette(dataset: VOSInferenceDataset):
    if dataset.name.startswith("youtube_vos"):
        fpath = osp.join(osp.dirname(__file__), "color_palette_ytvos.npy")
    elif dataset.name.startswith("davis") or dataset.name.startswith("HODOR"):
        fpath = osp.join(osp.dirname(__file__), "color_palette_davis.npy")
    else:
        raise ValueError(f"Invalid dataset name: '{dataset.name}'")

    return np.load(fpath)


def write_ytvos_zip(results_dir: str, dataset: VOSInferenceDataset, output_path: str):
    seq_names = sorted(os.listdir(results_dir))

    fout = zf.ZipFile(output_path, 'w', compression=zf.ZIP_DEFLATED)

    for seq_name in seq_names:
        eval_files = dataset.eval_files[seq_name]

        if eval_files:
            eval_files = [fname.replace(".jpg", ".png") for fname in eval_files]
        else:  # write all files to zip
            eval_files = sorted(os.listdir(osp.join(results_dir, seq_name)))

        for fname in eval_files:
            fout.write(osp.join(results_dir, seq_name, fname), arcname=osp.join("Annotations", seq_name, fname))

    fout.close()


def build_dataset(dataset_name: str, seq_names: Union[None, List[str]]):
    if dataset_name == "youtube_vos_val":
        dataset = VOSInferenceDataset(Paths.youtube_vos_val_images_dir(), Paths.youtube_vos_val_anns(),
                                      "youtube_vos_val", seq_names)

    elif dataset_name == "davis_val":
        dataset = VOSInferenceDataset(Paths.davis_trainval_images_dir(), Paths.davis_val_anns(), "davis_val", seq_names)

    elif dataset_name == "davis_testdev":
        dataset = VOSInferenceDataset(Paths.davis_testdev_images_dir(), Paths.davis_testdev_anns(), "davis_testdev",
                                      seq_names)

    else:
        raise ValueError(f"Invalid dataset specified: '{dataset_name}'")

    return dataset


def infer(model, dataset, output_dir, save_vis):
    data_loader = DataLoader(dataset, 1, False, num_workers=4, collate_fn=collate_fn_inference)

    cmap = load_color_palette(dataset)
    cmap_vis = np.flip(cmap, 1).tolist()
    cmap = cmap.flatten().tolist()

    total_frames = 0

    timer = Timer.get("inference")
    timer.tic()

    results_dir_name = "Annotations" if dataset.name.startswith("youtube_vos") else "results"

    for seq_num, sequence in tqdm(enumerate(data_loader), total=len(dataset), disable=False):
        meta_info = sequence['meta']

        seq_output_dir = osp.join(output_dir, results_dir_name, meta_info['seq_name'])
        vis_output_dir = osp.join(output_dir, "vis", meta_info['seq_name'])

        os.makedirs(seq_output_dir, exist_ok=True)

        if save_vis:
            os.makedirs(vis_output_dir, exist_ok=True)

        filenames = sequence["filenames"]

        images = sequence['images']  # [T, C, H, W]
        ref_masks = sequence['ref_masks']  # [N, H, W]
        ref_mask_frame_nums = sequence['ref_mask_frame_nums'].tolist()  # [N]
        instance_ids = sequence['instance_ids'].tolist()  # [N]

        total_frames += images.size(0)

        ref_instance_masks = defaultdict(list)
        for instance_id, first_frame_num, mask in zip(instance_ids, ref_mask_frame_nums, ref_masks):
            ref_instance_masks[first_frame_num].append((instance_id, mask))

        resized_dims = meta_info['resized_height'], meta_info['resized_width']
        orig_dims = meta_info['height'], meta_info['width']

        pred_seq_masks = model(
            frames=images, ref_masks_container=ref_instance_masks, resized_dims=resized_dims, orig_dims=orig_dims
        )

        timer.toc()
        if save_vis:
            images = reverse_image_scaling_and_normalization(
                images.cuda(), cfg.INPUT.IMAGE_MEAN, cfg.INPUT.IMAGE_STD, invert_channels=cfg.INPUT.RGB_INPUT,
                normalize_to_unit_scale=cfg.INPUT.NORMALIZE_TO_UNIT_SCALE
            )

            padding_h, padding_w = [o_dim - r_dim for r_dim, o_dim in zip(resized_dims, images.shape[-2:])]
            images = F.pad(images, (0, -padding_w, 0, -padding_h))
            images = F.interpolate(images, orig_dims, mode='bilinear', align_corners=False)
            images = images.permute(0, 2, 3, 1).round().byte().cpu().numpy()

        for t, (mask_t, image_t, fname_t) in enumerate(zip(pred_seq_masks, images, filenames)):
            mask_image = Image.fromarray(mask_t.cpu().numpy().astype(np.uint8))
            mask_image.putpalette(cmap)

            mask_image.save(osp.join(seq_output_dir, f"{fname_t}.png"))

            if save_vis:
                for iid in set(mask_t.unique().tolist()) - {0}:
                    mask_t_iid = (mask_t == iid).byte().cpu().numpy()
                    image_t = overlay_mask_on_image(image_t, mask_t_iid, mask_color=tuple(cmap_vis[iid]))

                cv2.imwrite(osp.join(vis_output_dir, f"{fname_t}.png"), image_t)
        timer.tic()

    timer.toc()
    total_duration = Timer.get_duration("inference")
    print(f"Total time: {total_duration} sec\n"
          f"Total frames: {total_frames}\n"
          f"FPS: {total_frames / total_duration}")

    if dataset.name.startswith("youtube_vos"):
        zip_path = osp.join(output_dir, "Annotations.zip")
        write_ytvos_zip(osp.join(output_dir, "Annotations"), dataset, zip_path)

    timer.reset()


def main(args):
    cfg_file = osp.join(osp.dirname(args.model_path), "config.yaml")

    assert osp.exists(cfg_file), "No config file found at expected path: {}".format(cfg_file)
    cfg.merge_from_file(cfg_file)

    if args.min_image_dim:
        ar = cfg.INPUT.MAX_DIM / float(cfg.INPUT.MIN_DIM)
        max_dim = int(round(args.min_image_dim * ar))
        cfg.INPUT.update_param("MIN_DIM", args.min_image_dim)
        cfg.INPUT.update_param("MAX_DIM", max_dim)
        print("Updated input dims to [{}, {}]".format(args.min_image_dim, max_dim))

    output_dir = args.output_dir
    if not osp.isabs(output_dir):
        output_dir = osp.join(osp.dirname(args.model_path), output_dir)

    model = build_model().cuda().eval()
    model.temporal_window = args.temporal_window

    # load trained weights
    with open(args.model_path, 'rb') as fh:
        model.load_state_dict(torch.load(fh)['model'], strict=True)

    dataset = build_dataset(args.dataset, args.seqs)

    print("Output directory: {}".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        infer(model, dataset, output_dir, args.save_vis)

    results_dir_name = "Annotations" if dataset.name.startswith("youtube_vos") else "results"
    print("Results written to: {}".format(osp.join(output_dir, results_dir_name)))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('model_path')

    parser.add_argument(
        '--dataset', '-d', required=True, choices=('davis_val', 'davis_testdev', 'youtube_vos_val'),
        help="Dataset on which to run inference."
    )
    parser.add_argument(
        '--output_dir', '-o', required=False,
        help="Path to output directory. Absolute and relative paths are both possible. Relative paths are appended to "
             "the directory containing the model checkpoint."
    )
    parser.add_argument(
        '--temporal_window', '-t', type=int, required=True,
        help="Number of frames to keep in the temporal history."
    )
    parser.add_argument(
        '--seqs', nargs="+", type=str, required=False, default=None,
        help="If inference has to be run only on certain video sequences, provide a list of the directory names of "
             "those videos here."
    )

    parser.add_argument(
        '--save_vis', action='store_true',
        help="If set, visualization of the results will also be written to disk."
    )
    parser.add_argument(
        '--min_image_dim', required=False, type=int,
        help="Set this argument if you want to infer using image dimensions different from the one used during training"
             ". This will overwrite the 'cfg.INPUT.MIN_DIM' parameter."
    )

    main(parser.parse_args())
