from torch.utils.data import ConcatDataset
from hodor.data.concat_dataset import ConcatDataset

from hodor.config import cfg
from hodor.data.images import CocoDataset
from hodor.data.vos import DenseVideoDataset, SparseVideoDataset
from hodor.utils.paths import Paths


def build_static_image_dataset(num_samples):
    dataset = CocoDataset(Paths.coco_train_images(), Paths.coco_train_anns(), num_samples, 
                          max_num_instances=cfg.TRAINING.NUM_INSTANCES,
                          resize_mode="px")

    return dataset


def build_sparse_video_dataset(num_samples, ds_config):
    assert abs(ds_config.YOUTUBE_VOS + ds_config.DAVIS - 1.0) < 1e-6

    ytvos_samples = int(round(num_samples * ds_config.YOUTUBE_VOS))
    davis_samples = num_samples - ytvos_samples

    datasets = []

    if ytvos_samples > 0:
        datasets.append(SparseVideoDataset(Paths.youtube_vos_train_images_dir(), Paths.youtube_vos_train_anns(),
                                           "YouTube-VOS", ytvos_samples,
                                           num_instances=cfg.TRAINING.NUM_INSTANCES,
                                           min_delta_t=cfg.TRAINING.MIN_CLIP_DELTA,
                                           max_delta_t=cfg.TRAINING.MAX_CLIP_DELTA,
                                           min_ref_mask_area=cfg.TRAINING.MIN_REF_MASK_AREA,
                                           min_ref_mask_dim=cfg.TRAINING.MIN_REF_MASK_DIM,
                                           apply_color_augmentations=cfg.TRAINING.AUGMENTATIONS.COLOR))

    if davis_samples > 0:
        datasets.append(SparseVideoDataset(Paths.davis_trainval_images_dir(), Paths.davis_train_anns(),
                                           "DAVIS", davis_samples,
                                           num_instances=cfg.TRAINING.NUM_INSTANCES,
                                           min_delta_t=cfg.TRAINING.MIN_CLIP_DELTA,
                                           max_delta_t=cfg.TRAINING.MAX_CLIP_DELTA * 2,
                                           min_ref_mask_area=cfg.TRAINING.MIN_REF_MASK_AREA,
                                           min_ref_mask_dim=cfg.TRAINING.MIN_REF_MASK_DIM,
                                           apply_color_augmentations=cfg.TRAINING.AUGMENTATIONS.COLOR))

    if len(datasets) > 1:
        return ConcatDataset(datasets)
    else:
        return datasets[0]


def build_dense_video_dataset(num_samples, ds_config):
    ytvos_share = ds_config.YTVOS
    assert 0. <= ytvos_share <= 1.0

    ytvos_num_samples = int(round(num_samples * ytvos_share))
    davis_num_samples = num_samples - ytvos_num_samples

    datasets = []

    if ytvos_num_samples > 0:
        datasets.append(DenseVideoDataset(
            Paths.youtube_vos_train_images_dir(), Paths.youtube_vos_train_anns(), "ytvos", ytvos_num_samples,
            num_instances=cfg.TRAINING.NUM_INSTANCES,
            min_delta_t=cfg.TRAINING.MIN_CLIP_DELTA,
            max_delta_t=cfg.TRAINING.MAX_CLIP_DELTA,
            min_ref_mask_area=cfg.TRAINING.MIN_REF_MASK_AREA,
            min_ref_mask_dim=cfg.TRAINING.MIN_REF_MASK_DIM,
            apply_color_augmentations=cfg.TRAINING.AUGMENTATIONS.COLOR
        ))

    if davis_num_samples > 0:
        datasets.append(DenseVideoDataset(
            Paths.davis_trainval_images_dir(), Paths.davis_train_anns(), "davis", davis_num_samples,
            num_instances=cfg.TRAINING.NUM_INSTANCES,
            min_delta_t=cfg.TRAINING.MIN_CLIP_DELTA,
            max_delta_t=cfg.TRAINING.MAX_CLIP_DELTA * 2,
            min_ref_mask_area=cfg.TRAINING.MIN_REF_MASK_AREA,
            min_ref_mask_dim=cfg.TRAINING.MIN_REF_MASK_DIM,
            apply_color_augmentations=cfg.TRAINING.AUGMENTATIONS.COLOR
        ))

    if len(datasets) > 1:
        return ConcatDataset(datasets)
    else:
        return datasets[0]
