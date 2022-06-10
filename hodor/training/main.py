from argparse import ArgumentParser
from collections import OrderedDict
from datetime import timedelta
from multiprocessing import cpu_count
from torch.utils.data import DataLoader

from hodor.config import cfg
from hodor.data.utils.preprocessing import collate_fn_train
from hodor.training.train_model import TrainModel
from hodor.utils.paths import Paths
from hodor.training.batch_sampler import CustomBatchSampler
from hodor.training.optim_utils import build_lr_scheduler
from hodor.training.pred_mask_manager import PredMaskManager
from hodor.training.dice_loss import multiclass_dice_loss

from hodor.training import utils as utils
from hodor.training.distributed_sampler import DistributedSampler
from hodor.training.trainer_base import TrainerBase, InterruptException


import cv2
import os
import imgaug as ia
import numpy as np
import os.path as osp
import logging
import random
import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
import torch.nn.functional as F

import hodor.training.data_utils as dataset_utils


class Trainer(TrainerBase):
    """

    """
    def __init__(self, model_save_dir, args):
        model = TrainModel()
        super(Trainer, self).__init__(model, model_save_dir)
        model.logger = self.console_logger

        optimizer = optim.AdamW(self.model.parameters(), cfg.TRAINING.BASE_LR, weight_decay=0.01)
        lr_scheduler = build_lr_scheduler(optimizer, cfg.TRAINING.LR_SCHEDULER)

        if lr_scheduler is None:
            self.register_optimizer(optimizer)
        else:
            self.register_optimizer(optimizer, [lr_scheduler])

        self.post_init(
            args.restore_session, args.restore_path, mixed_precision=False, collate_fn=collate_fn_train,
            restoration_strict=True, find_unused_parameters=True, convert_sync_batch_norm=False
        )
        self.detect_anomaly = args.detect_anomaly

        self.dataset_mode = cfg.TRAINING.MODE
        self.ignore_oom_errors = not args.raise_oom_errors
        self.pin_dataloader_memory = args.pin_memory

    def create_training_dataset(self, total_samples: int):
        if self.dataset_mode == "static_image":
            return dataset_utils.build_static_image_dataset(total_samples)

        elif self.dataset_mode == "sparse_video":
            return dataset_utils.build_sparse_video_dataset(total_samples, cfg.DATA.SPARSE_VIDEO_SPLIT)

        elif self.dataset_mode == "dense_video":
            return dataset_utils.build_dense_video_dataset(total_samples, cfg.DATA.DENSE_VIDEO_SPLIT)

        else:
            raise ValueError(f"Invalid training mode argument provided: '{self.dataset_mode}'")

    def create_training_data_loader(self, dataset, sub_iter_batch_size_per_gpu: int, batch_size: int,
                                    total_iterations: int, optimizer_step_interval: int,
                                    num_workers: int):

        num_samples = batch_size * total_iterations
        assert len(dataset) == num_samples, f"Length mismatch: {len(dataset)} =/= {num_samples}"

        if utils.is_distributed():
            sampler = DistributedSampler(dataset, self.num_gpus, self.local_rank, shuffle=True,)
        else:
            sampler = None

        if num_workers < 0:
            num_workers = min(max(cpu_count() // self.num_gpus, 1), 8)

        batch_sampler = CustomBatchSampler(sampler, dataset, sub_iter_batch_size_per_gpu,
                                           post_shuffle=True, max_allowed_ar_diff=0.1,
                                           elapsed_batches=self.elapsed_iterations)
        sampler = None
        shuffle = False
        loader_batch_size = 1

        return DataLoader(dataset, loader_batch_size, shuffle, sampler, batch_sampler=batch_sampler,
                          num_workers=num_workers, collate_fn=self.collate_fn, timeout=30, prefetch_factor=2,
                          pin_memory=self.pin_dataloader_memory, worker_init_fn=dataloder_worker_init)

    def forward(self, training_sample, subiter):
        training_sample = self.torch_to(training_sample, device=self.local_device)

        # ------------- Filter out instances whose masks will become all zeros after downsampling to 8x ---------------
        scale_factor = 1. / self._model.encoder.mask_scale
        ref_mask = F.interpolate(training_sample["ref_mask"][None].byte(), scale_factor=scale_factor, mode='nearest',
                                 recompute_scale_factor=False)[0].long()

        unique_vals = [ref_mask[b].unique().tolist() for b in range(ref_mask.size(0))]
        num_instances = training_sample["num_instances"]
        valid_ids = set()

        for iid in range(1, num_instances + 1):
            if all([iid in iids_per_sample for iids_per_sample in unique_vals]):
                valid_ids.add(iid)

        if len(valid_ids) == 0:
            print(f"ERROR: All instances ({num_instances}) have null masks after downsampling")
            valid_ids.add(1)

        elif len(valid_ids) != num_instances > 0:
            print(f"Not all instance IDs have valid downsampled masks: valid IDs: {valid_ids}, no. of instances: "
                  f"{num_instances}")

        if len(valid_ids) == num_instances:
            updated_ref_mask = training_sample["ref_mask"]
            updated_other_masks = training_sample["other_masks"]
            updated_presence_map = training_sample["instance_presence_map"]

        else:
            updated_ref_mask = torch.zeros_like(training_sample["ref_mask"])

            if training_sample["other_masks"] is None:
                updated_other_masks = None
            else:
                updated_other_masks = torch.zeros_like(training_sample["other_masks"])

            for new_iid, iid in enumerate(valid_ids, 1):
                updated_ref_mask = torch.where(training_sample["ref_mask"] == iid, 
                                               torch.full_like(updated_ref_mask, new_iid), 
                                               updated_ref_mask)

                if updated_other_masks is not None:
                    updated_other_masks = torch.where(training_sample["other_masks"] == iid,
                                                      torch.full_like(updated_other_masks, new_iid),
                                                      updated_other_masks)

            presence_map = training_sample["instance_presence_map"]  # [B, T, I]
            idxes = sorted([iid - 1 for iid in valid_ids])
            updated_presence_map = presence_map.permute(2, 0, 1)[idxes]  # [I', B, T]
            updated_presence_map = updated_presence_map.permute(1, 2, 0)  # [B, T, I']

        pred_masks_manager = PredMaskManager.from_tensors(
            updated_presence_map, updated_ref_mask, updated_other_masks
        )
        # -------------------------------------------------------------------------------------

        output_dict = self.model(
            ref_frame=training_sample['ref_image'],
            ref_mask=updated_ref_mask,
            other_frames=training_sample['other_images'],
            iter_num=self.elapsed_iterations + subiter,
            num_instances=len(valid_ids),
            pred_masks_manager=pred_masks_manager
        )

        return PredMaskManager.from_dict(output_dict)

    def apply_criterion(self, model_output, training_sample):
        assert isinstance(model_output, PredMaskManager)
        pred_mask_manager = model_output

        if cfg.TRAINING.APPLY_LOSS_ON_INTERMEDIATE_FRAMES:
            pred_mask_logits, gt_masks = pred_mask_manager.get_all_pred_gt_pairs(condense_gt_mask=True)
        else:
            pred_mask_logits, gt_masks = pred_mask_manager.get_pred_gt_pairs(frame_id=0, condense_gt_mask=True)

        gt_masks = gt_masks.detach()
        loss = multiclass_dice_loss(pred_mask_logits, gt_masks) + (1.0 * F.cross_entropy(pred_mask_logits, gt_masks))

        return {
            "loss": loss
        }

    def compute_loss_scalar(self, model_output, training_sample, criterion_output):
        return criterion_output["loss"]

    @torch.no_grad()
    def get_log_vars(self, loss_scalar, training_sample, model_output, criterion_output):
        assert isinstance(model_output, PredMaskManager)
        pred_mask_manager = model_output

        vars_dict = {
            "Loss": loss_scalar,
            "AllocM": torch.as_tensor(
                utils.get_allocated_memory(loss_scalar, unit="GB") + utils.get_cached_memory(loss_scalar, unit="GB"),
                device=self.local_device
            )
        }

        pred_mask_logits, gt_masks = pred_mask_manager.get_all_pred_gt_pairs(condense_gt_mask=True)
        pred_masks = pred_mask_logits.argmax(1)  # [B, N, H, W]
        assert pred_masks.shape == gt_masks.shape, f"Shape mismatch: {pred_masks.shape}, {gt_masks.shape}"

        inter = 0.
        union = 0.

        for instance_id in range(1, pred_mask_manager.num_instances + 1):
            pred_iid = pred_masks == instance_id
            gt_iid = gt_masks == instance_id

            inter += torch.logical_and(pred_iid, gt_iid).sum(dtype=torch.long)
            union += torch.logical_or(pred_iid, gt_iid).sum(dtype=torch.long)

        if union == 0:
            self.console_logger("Union of predicted gt masks is zero!")
            union = 1

        vars_dict["IoU"] = inter / union

        return vars_dict

    def get_pretraining_print_list(self):
        return OrderedDict([
            ("Mode",                                        self.dataset_mode),
            ("Transformer Pre-normalize",                   cfg.MODEL.TRANSFORMER_PRE_NORMALIZE),
            ("No. of background descriptors",               cfg.MODEL.NUM_BG_DESCRIPTORS),
            ("No. of catch all queries",                    cfg.MODEL.NUM_CATCH_ALL_QUERIES),
            ("No. of Attention Layers",                     (cfg.MODEL.ENCODER_LAYERS, cfg.MODEL.DECODER_LAYERS)),
            ("Positional Encodings (FMQ, FQM)",             (cfg.MODEL.POS_ENCODINGS_IN_FMQ, cfg.MODEL.POS_ENCODINGS_IN_FQM)),
            ("Input Dims",                                  (cfg.INPUT.MIN_DIM, cfg.INPUT.MAX_DIM)),
            ("Intermediate Masks Detached",                 cfg.TRAINING.DETACH_INTERMEDIATE_MASKS),
            ("No. of Other Frames",                         cfg.TRAINING.NUM_OTHER_FRAMES),
            ("No. of Instances",                            cfg.TRAINING.NUM_INSTANCES)
        ])

    def get_oom_error_extra_details(self, training_sample):
        info = "Resized dims: {}\n".format(list(training_sample["ref_mask"].shape[-2:]))

        info += "Sample details: "
        for meta in training_sample["meta"]:
            info += "[dataset={}, sequence={}, dims=({}, {})], ".format(
                meta["dataset"], meta["seq_id"], meta["height"], meta["width"]
            )

        return info

    @classmethod
    def torch_to(cls, x, *args, **kwargs):
        if torch.is_tensor(x):
            return x.to(*args, **kwargs)
        elif isinstance(x, (list, tuple)):
            return type(x)([cls.torch_to(elem, *args, **kwargs) for elem in x])
        elif isinstance(x, dict):
            return {k: cls.torch_to(v, *args, **kwargs) for k, v in x.items()}
        elif hasattr(x, "to"):
            return x.to(*args, **kwargs)
        else:
            return x


def dataloder_worker_init(worker_id: int) -> None:
    seed = 220294 + worker_id + utils.get_rank()

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ia.seed(seed)


def parse_log_level(s):
    s = s.lower()
    if s == "info":
        return logging.INFO
    elif s == "debug":
        return logging.DEBUG
    elif s == "warn":
        return logging.WARN
    elif s == "error":
        return logging.ERROR
    elif s == "fatal":
        return logging.FATAL
    else:
        raise ValueError("Invalid string: '{}'".format(s))


def start(args):
    # fix random seeds
    torch.manual_seed(42)
    random.seed(42 + utils.get_rank())
    np.random.seed(42 + utils.get_rank())
    ia.seed(42 + utils.get_rank())

    if utils.is_main_process():
        log_level = parse_log_level(args.log_level)
    else:
        log_level = parse_log_level(args.subprocess_log_level)

    if osp.isabs(args.model_dir):
        model_save_dir = args.model_dir
    else:
        model_save_dir = osp.join(Paths.saved_models(), args.model_dir)

    trainer = Trainer(model_save_dir, args)
    trainer.console_logger.setLevel(log_level)

    # backup config to model directory
    if utils.is_main_process():
        with open(osp.join(model_save_dir, 'config.yaml'), 'w') as writefile:
            yaml.dump(cfg.d(), writefile)

    cfgt = cfg.TRAINING
    trainer.start(total_iterations=cfgt.TOTAL_ITERATIONS,
                  batch_size=cfgt.BATCH_SIZE,
                  accumulate_gradients=cfgt.ACCUMULATE_GRADIENTS,
                  clip_gradients=False,
                  max_samples_per_gpu=args.max_samples_per_gpu,
                  data_loader_cpu_workers=args.num_cpu_workers,
                  display_interval=args.display_interval,
                  summary_interval=args.summary_interval,
                  save_interval=args.save_interval,
                  save_ckpts_after=args.save_ckpts_after)


def init_distributed(args, num_gpus):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port if args.master_port else '12356'

    # initialize the process group
    timeout = timedelta(0, 25)  # 25 seconds
    dist.init_process_group("nccl", rank=args.local_rank, world_size=num_gpus, timeout=timeout)

    try:
        with torch.cuda.device(args.local_rank):
            start(args)
    except InterruptException as _:
        print("Training session was interrupted")

    utils.synchronize()
    dist.destroy_process_group()


def main(args):
    if args.cv2_num_threads:
        assert args.cv2_num_threads >= 0
        cv2.setNumThreads(args.cv2_num_threads)

    cfg_path = None

    if args.restore_session:
        expected_cfg_path = os.path.realpath(os.path.join(args.restore_session, os.pardir, 'config.yaml'))
        if os.path.exists(expected_cfg_path):
            cfg_path = expected_cfg_path
        else:
            print("[ WARN] No config file was found in the directory containing the restore checkpoint.")

    else:
        assert args.cfg, f"'--cfg' argument is required if not restoring session."
        if osp.isabs(args.cfg):
            cfg_path = args.cfg
        else:
            cfg_path = osp.join(Paths.configs_dir(), args.cfg)

    if cfg_path:
        print("Restoring config from: {}".format(cfg_path))
        cfg.merge_from_file(cfg_path)

    cfg.update_from_args(args, verbose=True)

    num_gpus = torch.cuda.device_count()
    if args.allow_multigpu and num_gpus > 1:
        init_distributed(args, num_gpus)
    else:
        start(args)


if __name__ == "__main__":
    parser = ArgumentParser()

    param_group_main = parser.add_argument_group("Main options")

    param_group_main.add_argument(
        '--cfg', required=False,
        help="Path to config file. Absolute and relative paths can be given. Relative paths are appended to 'configs/'."
             "This argument is required unless '--restore_session' is set."
    )

    param_group_main.add_argument(
        '--model_dir', type=str, required=True,
        help="Directory where the model checkpoint and log files will be saved. Relative paths will be appended to "
             "'$HODOR_WORKSPACE_DIR/checkpoints'"
    )
    param_group_main.add_argument(
        '--restore_session', type=str, required=False,
        help="To resume training, point this to the .pth checkpoint from where you want to resume."
    )
    param_group_main.add_argument(
        '--restore_path', type=str, required=False,
        help="To start training from a pretrained checkpoint, point this to the .pth of the checkpoint. Note that this"
             " is different from '--restore_session'. This will not resume training from the given checkpoint, but "
             "rather just restore these weights and then start a fresh training.")

    # Options related to multi-GPU training
    param_group_multigpu = parser.add_argument_group("Multi-GPU related options")

    param_group_multigpu.add_argument(
        '--local_rank', type=int, default=0,
        help="Do not set this manually. This argument is automatically passed to this script when running with in "
             "multi-GPU mode."
    )
    param_group_multigpu.add_argument(
        '--master_port', type=str, default='12356'
    )
    param_group_multigpu.add_argument(
        '--allow_multigpu', action='store_true',
        help="If this is not set, training will run on a single GPU even if multiple are available on the machine"
    )
    param_group_multigpu.add_argument(
        '--max_samples_per_gpu', type=int, default=2,
        help="Number of training samples to run on a single GPU. On a GPU with 24GB VRAM it's possible to run 2. If "
             "your GPU has lesser VRAM try running with 1."
    )
    param_group_multigpu.add_argument(
        '--num_cpu_workers', type=int, default=-1,
        help="Number of worker processes used by the dataloader. By default it will be set to the number of available "
             "CPU cores on the machine, up to a maximum value of 8."
    )
    param_group_multigpu.add_argument(
        '--cv2_num_threads', type=int, required=False,
        help="Number of CPU threads used internally by OpenCV. Leave this unset unless you encounter strange errors "
             "in the data loading with cv2 functions."
    )

    param_group_logging = parser.add_argument_group("Logging/checkpointing related options")

    param_group_logging.add_argument(
        '--display_interval', type=int, default=5,
        help="Number of iterations after which stats are printed to the console screen."
    )
    param_group_logging.add_argument(
        '--summary_interval', type=int, default=10,
        help="Number of iterations after which a TensorboardX summary is written to disk."
    )
    param_group_logging.add_argument(
        '--save_interval', type=int, default=10000,
        help="Number of iterations after which a model checkpoint is written to disk."
    )
    param_group_logging.add_argument(
        '--save_ckpts_after', type=int, default=0,
        help="Number of iterations after which checkpoint saving begins. E.g. if this is set to 50,000 and "
             "'--save_interval' is set to 10,000 then checkpoints will be saved at 50k, 60k, 70k.... iterations."
    )
    param_group_logging.add_argument(
        '--log_level',  type=str, default="info",
        help="Logging level for the main process."
    )
    param_group_logging.add_argument(
        '--subprocess_log_level', type=str, default="warn",
        help="Logging level for the subprocesses when running in multi-GPU mode."
    )

    param_group_misc = parser.add_argument_group("Miscellaneous options")

    param_group_misc.add_argument(
        '--pin_memory', action='store_true',
        help="Sets the 'pin_memory' option for the data loader."
    )
    param_group_misc.add_argument(
        '--detect_anomaly', action='store_true',
        help="If set, training will run with 'torch.autograd.set_detect_anomaly(True)'. This is helpful mainly in "
             "pinpointing where in the code NaN values are first generated."
    )
    param_group_misc.add_argument(
        '--raise_oom_errors', action='store_true',
        help="If set, trainer will not attempt to recover from OOM errors."
    )

    parser = cfg.add_args_to_parser(parser)

    main(parser.parse_args())
