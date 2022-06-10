from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, List, Optional, Any, Callable
from collections import defaultdict, OrderedDict
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from torch import Tensor
from time import time as current_time

from hodor.training.tensorboardx_logger import TrainingLogger
from hodor.training import distributed_utils as dist_utils

import contextlib
import logging
import os
import os.path as osp
import signal
import torch
import torch.nn as nn

try:
    import apex.amp as amp

    APEX_IMPORTED = True
except ImportError as _:
    print("NOTE: Could not import apex. Mixed precision training will be unavailable.")
    APEX_IMPORTED = False


def create_logger(main_log_level, subprocess_log_level, file_output_path=None):
    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(proc_id)d] %(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
    ch.setFormatter(formatter)

    if dist_utils.is_main_process():
        ch.setLevel(main_log_level)
    else:
        ch.setLevel(subprocess_log_level)

    logger.addHandler(ch)

    if file_output_path is not None:
        fh = logging.FileHandler(file_output_path, 'w')
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False

    extra = {"proc_id": dist_utils.get_rank()}

    logger = logging.LoggerAdapter(logger, extra)
    logger.propagate = False

    return logger


class InterruptException(RuntimeError):
    pass


class InterruptDetector:
    def __init__(self):
        self.__is_interrupted = False

    def start(self):
        signal.signal(signal.SIGINT, self.__set_interrupted)
        signal.signal(signal.SIGTERM, self.__set_interrupted)

    def __set_interrupted(self, signum, frame):
        self.__is_interrupted = True

    is_interrupted = property(fget=lambda self: self.__is_interrupted)


@contextlib.contextmanager
def timed_event(print_fn: Callable, label: str, enable: bool):
    start_time = current_time()
    yield None
    end_time = current_time()
    if enable:
        print_fn(f"{label}: {end_time - start_time:.3f} sec")


class TrainerBase(metaclass=ABCMeta):
    def __init__(self, model, model_save_dir, logger=None):
        self.num_gpus = dist_utils.get_world_size()
        self.local_rank = dist_utils.get_rank()
        self.local_device = dist_utils.get_device()
        self.is_main_process = dist_utils.is_main_process()

        self.model = model
        self.model_save_dir = model_save_dir
        self.log_dir = osp.join(self.model_save_dir, 'logs')

        if self.is_main_process:
            os.makedirs(self.log_dir, exist_ok=True)

        if logger is None:
            if self.is_main_process:
                log_txt_file = osp.join(self.log_dir, "out.log")
            else:
                log_txt_file = None

            self.console_logger = create_logger(logging.INFO, logging.WARN, file_output_path=log_txt_file)
        else:
            self.console_logger = logger

        self.optimizers_and_lr_schedulers = []

        # create parameter logger
        self.logger = None
        if self.is_main_process:
            self.logger = TrainingLogger(self.log_dir)

        self.interrupt_detector = InterruptDetector()

        self.elapsed_iterations = 0

        self.detect_anomaly = False
        self.ignore_oom_errors = False
        self.log_time_durations = False

        self.collate_fn = None
        self.__mp_enabled = False
        self.__post_init_called = False

        self.display_interval = -1
        self.summary_interval = -1
        self.save_interval = -1

    def register_optimizer(self, optimizer: torch.optim.Optimizer,
                           lr_schedulers: Optional[List[torch.optim.lr_scheduler._LRScheduler]] = None):
        assert isinstance(lr_schedulers, list) or lr_schedulers is None
        self.optimizers_and_lr_schedulers.append([optimizer, lr_schedulers])

    def post_init(self, restore_session: Optional[str] = None, restore_path: Optional[str] = None,
                  mixed_precision: bool = False, collate_fn: Optional[Callable] = None,
                  find_unused_parameters: bool = True, **kwargs):

        if len(self.optimizers_and_lr_schedulers) == 0:
            raise ValueError("At least one optimizer has to be registered before calling 'post_init'")

        mixed_precision_level = kwargs.get("mp_level", "O1")
        restoration_strict = kwargs.get("restoration_strict", True)

        self.model = self.model.to(self.local_device)
        self.collate_fn = collate_fn

        if mixed_precision:
            assert APEX_IMPORTED
            self.console_logger.info("Mixed precision training is enabled.")

            optimizers = [entry[0] for entry in self.optimizers_and_lr_schedulers]
            self.model, optimizers = amp.initialize(self.model, optimizers, opt_level=mixed_precision_level)

            for i in range(len(self.optimizers_and_lr_schedulers)):
                self.optimizers_and_lr_schedulers[i][0] = optimizers[i]

            self.__mp_enabled = True

        if dist_utils.is_distributed():
            if kwargs.get("convert_sync_batch_norm", False):
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=find_unused_parameters
            )

        assert not (restore_session and restore_path)

        if restore_session:
            self.console_logger.info(f"Restoring session from {restore_session}")
            self.restore_session(torch.load(restore_session, map_location=self.local_device))
        elif restore_path:
            self.console_logger.info(f"Restoring saved model from {restore_path}")
            self.restore_path(restore_path, strict=restoration_strict)

        self.__post_init_called = True

    @property
    def _model(self):
        return self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

    @abstractmethod
    def create_training_dataset(self, total_samples: int) -> Dataset:
        pass

    @abstractmethod
    def create_training_data_loader(self, dataset: Dataset, sub_iter_batch_size_per_gpu: int, batch_size: int,
                                    total_iterations: int, optimizer_step_interval: int,
                                    num_workers: int) -> DataLoader:
        pass

    @abstractmethod
    def forward(self, training_sample: Any, subiter: int) -> Any:
        pass

    @abstractmethod
    def apply_criterion(self, model_output: Any, training_sample: Any) -> Any:
        pass

    @abstractmethod
    def compute_loss_scalar(self, model_output, training_sample, criterion_output) -> Tensor:
        pass

    @abstractmethod
    def get_log_vars(self, loss_scalar: Tensor, training_sample: Any, model_output: Any, criterion_output) -> Dict[str, Tensor]:
        pass

    def backup_session(self):
        model_save_path = osp.join(self.model_save_dir, f"{self.elapsed_iterations:06d}.pth")

        optimizers_and_lr_schedulers_state_dicts = [
            (opt.state_dict(), [lrs.state_dict() for lrs in lr_schedulers] if lr_schedulers is not None else None)
            for opt, lr_schedulers in self.optimizers_and_lr_schedulers
        ]

        save_dict = {
            'model': self._model.state_dict(),
            'optimizers_and_lr_schedulers': optimizers_and_lr_schedulers_state_dicts,
            'logger': self.logger.state_dict(),
            'iterations': self.elapsed_iterations
        }

        if self.__mp_enabled:
            save_dict['amp'] = amp.state_dict()

        torch.save(save_dict, model_save_path)
        self.console_logger.info(f"Checkpoint saved to: {model_save_path}")
        return model_save_path

    def restore_path(self, restore_path: str, strict: bool):
        return self._model.load_state_dict(torch.load(restore_path, map_location=self.local_device)['model'],
                                           strict=strict)

    def restore_session(self, restore_dict: Dict):
        assert 'model' in restore_dict, "Restore state dict contains no entry named 'model'"
        self._model.load_state_dict(restore_dict['model'])

        assert 'optimizers_and_lr_schedulers' in restore_dict, \
            "Restore state dict contains no entry named 'optimizers_and_lr_schedulers'"

        assert len(restore_dict['optimizers_and_lr_schedulers']) == len(self.optimizers_and_lr_schedulers), \
            "Length mismatch: {}, {}".format(
                len(restore_dict['optimizers_and_lr_schedulers']), len(self.optimizers_and_lr_schedulers)
            )

        for i, (opt_state_dict, lr_state_dicts) in enumerate(restore_dict['optimizers_and_lr_schedulers']):
            self.optimizers_and_lr_schedulers[i][0].load_state_dict(opt_state_dict)
            if lr_state_dicts is not None:
                for j, lrs_dict in enumerate(lr_state_dicts):
                    self.optimizers_and_lr_schedulers[i][1][j].load_state_dict(lrs_dict)

        if self.is_main_process:
            assert 'logger' in restore_dict, "Restore state dict contains no entry named 'logger'"
            self.logger.load_state_dict(restore_dict['logger'])

        assert 'iterations' in restore_dict, "Restore state dict contains no entry named 'iterations'"
        self.elapsed_iterations = restore_dict['iterations']

        if self.__mp_enabled:
            amp.load_state_dict(restore_dict['amp'])

    def backward(self, loss_scalar: Tensor, training_sample: Any, model_output: Any) -> None:
        if self.__mp_enabled:
            for opt, _ in self.optimizers_and_lr_schedulers:
                with amp.scale_loss(loss_scalar, opt) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss_scalar.backward()

    def calculate_optimizer_step_parameters(self, accumulate_gradients: bool, batch_size: int,
                                            max_samples_per_gpu: int) -> Tuple[int, int]:
        if accumulate_gradients:
            # ensure that batch size is larger than the number of available GPUs
            assert batch_size >= self.num_gpus, f"Batch size ({batch_size}) must be >= number of GPUs ({self.num_gpus})"

            if batch_size < (max_samples_per_gpu * self.num_gpus):
                # we have more GPUs than needed (Yay!)
                assert batch_size % self.num_gpus == 0, \
                    f"Batch size ({batch_size}) must be exactly divisible by number of GPUs ({self.num_gpus})"
                optimizer_step_interval = 1
            else:
                assert batch_size % min(batch_size, max_samples_per_gpu) == 0
                optimizer_step_interval = int(batch_size / (min(batch_size, max_samples_per_gpu) * self.num_gpus))

            assert optimizer_step_interval > 0, \
                f"Oops! Something went wrong. Given params: batch_size={batch_size}, " \
                    f"max_samples_per_gpu={max_samples_per_gpu}, num_gpus={self.num_gpus}"

            self.console_logger.info(f"Optimizer will be run every {optimizer_step_interval} iterations")

        else:
            if batch_size > (self.num_gpus * max_samples_per_gpu):
                raise ValueError(
                    f"A batch size of {batch_size} cannot be achieved because max "
                    f"samples per GPU = {max_samples_per_gpu} and num GPUs = {self.num_gpus} (product of the two is "
                    f"less than batch size)"
                )
            optimizer_step_interval = 1

        sub_iter_batch_size_per_gpu = batch_size // (optimizer_step_interval * self.num_gpus)

        assert 0 < sub_iter_batch_size_per_gpu <= batch_size, \
            "Oops! Something went wrong. Given params: batch_size={}, max_samples_per_gpu={}, num_gpus={}," \
            "optimizer_step_interval={}".format(
                batch_size, max_samples_per_gpu, self.num_gpus, optimizer_step_interval)

        return sub_iter_batch_size_per_gpu, optimizer_step_interval

    def start(self,
              total_iterations: int,
              batch_size: int,
              accumulate_gradients: bool,
              clip_gradients: bool,
              max_samples_per_gpu: int,
              display_interval: int,
              summary_interval: int,
              save_interval: int,
              save_ckpts_after: int = 0,
              data_loader_cpu_workers: int = -1):

        sub_iter_batch_size_per_gpu, optimizer_step_interval = self.calculate_optimizer_step_parameters(
            accumulate_gradients, batch_size, max_samples_per_gpu
        )

        self.display_interval = display_interval
        self.summary_interval = summary_interval
        self.save_interval = save_interval

        print_list = OrderedDict([
            ("Elapsed iterations", self.elapsed_iterations),
            ("Total iterations", total_iterations),
            ("Batch size", batch_size),
            ("Sub-iteration batch size per GPU", sub_iter_batch_size_per_gpu),
            ("Optimizer step interval", optimizer_step_interval),
            ("Model save directory", self.model_save_dir),
            ("Save interval", save_interval),
            ("Gradient clipping", "Enabled" if clip_gradients else "Disabled"),
            ("Trainable parameters", sum([p.numel() for p in self._model.parameters() if p.requires_grad]))
        ])
        print_list.update(self.get_pretraining_print_list())
        print_list = "\n".join(["- {}: {}".format(name, val) for name, val in print_list.items()])

        self.console_logger.info(f"Commencing/resuming training with the following settings:\n{print_list}")

        dataset = self.create_training_dataset(batch_size * total_iterations)
        data_loader = self.create_training_data_loader(dataset, sub_iter_batch_size_per_gpu, batch_size,
                                                       total_iterations,
                                                       optimizer_step_interval, data_loader_cpu_workers)
        data_loader = iter(data_loader)

        if self.is_main_process:
            self.logger.total_iterations = total_iterations
            self.logger.start_timer()

        dist_utils.synchronize()

        self.interrupt_detector.start()  # useful when running with SLURM.

        step_idx = 0
        log_vars_cache = defaultdict(lambda: 0.0)

        # Computing logging metrics can be time consuming, so only do it when need (for display, summary etc.)
        logging_metrics_needed = (self.elapsed_iterations + 1) % display_interval == 0 or \
                                 (self.elapsed_iterations + 1) % summary_interval == 0

        while True:
            try:
                with timed_event(self.console_logger.info, "Training sample generation duration",
                                 enable=self.log_time_durations):
                    training_sample = next(data_loader)
            except StopIteration as _:  # this is raised when the data loader has finished all its samples
                break

            with torch.autograd.set_detect_anomaly(self.detect_anomaly):
                # (1) Forward pass through the model
                try:
                    with timed_event(self.console_logger.info, label="Forward pass duration",
                                     enable=self.log_time_durations):
                        model_output = self.forward(training_sample, step_idx)

                except RuntimeError as exc:
                    # Try to recover from an out-of-memory error if not running on multi-GPU
                    if self.is_oom_error(repr(exc)):
                        print_str = f"OOM occurred during forward pass in process rank={self.local_rank} at " \
                            f"iter={self.elapsed_iterations + 1}, sub_iter={step_idx + 1}."

                        extra_details = self.get_oom_error_extra_details(training_sample=training_sample)
                        if extra_details:
                            print_str += f"\nExtra details: {extra_details}"

                        self.console_logger.error(print_str)

                        if self.ignore_oom_errors and not dist_utils.is_distributed():
                            torch.cuda.empty_cache()
                            continue

                    raise exc

                dist_utils.synchronize()
                if self.interrupt_detector.is_interrupted:
                    raise InterruptException()

                # (2) Calculate loss
                criterion_output = self.apply_criterion(model_output, training_sample)
                loss = self.compute_loss_scalar(model_output, training_sample, criterion_output)
                assert loss.numel() == 1, f"Final loss must be a scalar value, but got tensor of shape {list(loss.shape)}"

                # (3) Backprop loss. Again, try to recover from an OOM error if one is raised
                try:
                    with timed_event(self.console_logger.info, label="Backward pass duration",
                                     enable=self.log_time_durations):
                        self.backward(loss / float(optimizer_step_interval), training_sample, model_output)

                except RuntimeError as exc:
                    if self.ignore_oom_errors and self.is_oom_error(repr(exc)) and not dist_utils.is_distributed():
                        self.console_logger.error("OOM error occurred during backward pass, but will be ignored.")
                        for opt, _ in self.optimizers_and_lr_schedulers:
                            opt.zero_grad()
                        step_idx = 0
                        del model_output, criterion_output, loss
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise exc

            # (4) Get values that need to be logged in Tensorboard and/or displayed
            with torch.no_grad():
                if logging_metrics_needed:
                    log_vars = self.get_log_vars(loss, training_sample, model_output, criterion_output)
                    for k, v in log_vars.items():
                        assert torch.is_tensor(v), f"Log variables must be of type torch.Tensor, but got '{type(v)}'"
                        assert v.ndim == 0, f"Logging variables should be scalar values, but got tensor of " \
                            f"shape {list(v.shape)} for key '{k}'"
                        log_vars_cache[k] += (v / float(optimizer_step_interval))

            # When accumulating gradients, we only perform the parameter update (or `step`) every
            # `optimizer_step_interval` iterations.
            step_idx += 1
            if step_idx < optimizer_step_interval:
                continue
            step_idx = 0

            # (5) Clip gradients if needed
            if clip_gradients:
                gradient_norm = clip_grad_norm_(self._model.parameters(), self.get_gradient_clip_value())
                self.console_logger.debug("Gradient norm: {}".format(gradient_norm))

                if gradient_norm != gradient_norm:  # NaN gradients
                    self.console_logger.warn("Gradient norm is NaN. Zeroing all gradients")
                    for opt, _ in self.optimizers_and_lr_schedulers:
                        opt.zero_grad()

            # (6) Perform `step` i.e. model parameter update. Also advance the LR scheduler
            with timed_event(self.console_logger.info, "Optimizer/LRScheduler step duration",
                             enable=self.log_time_durations):
                for opt, _ in self.optimizers_and_lr_schedulers:
                    opt.step()

                # first call 'step' on all learning rate schedulers
                for _, lr_schedulers in self.optimizers_and_lr_schedulers:
                    if lr_schedulers is not None:
                        for lrs in lr_schedulers:
                            lrs.step()

                # then call 'step' on all optimizers
                for opt, _ in self.optimizers_and_lr_schedulers:
                    opt.zero_grad()

            self.elapsed_iterations += 1

            # (7) This step involves displaying the logged values. Under multi-GPU the values need to be averaged
            # across all processes.
            if logging_metrics_needed:
                self.post_step_routine(training_sample, model_output, log_vars_cache)

            log_vars_cache.clear()

            # (8) Save checkpoint if needed
            if self.elapsed_iterations % save_interval == 0 and self.elapsed_iterations > save_ckpts_after:
                if self.is_main_process:
                    self.backup_session()

                dist_utils.synchronize()

            # determine if metrics need to be logged for the next iteration
            logging_metrics_needed = (self.elapsed_iterations + 1) % display_interval == 0 or \
                                     (self.elapsed_iterations + 1) % summary_interval == 0

            del model_output, criterion_output, loss

        self.console_logger.info(
            f"Training complete\n"
            f"Model(s) saved to: {self.model_save_dir}\n"
            f"Log file(s) saved to: {self.log_dir}\n"
        )

    def post_step_routine(self, training_sample, model_output, log_vars_cache):
        logging_vars = dist_utils.reduce_dict(dict(log_vars_cache), average=True)
        logging_vars_str = " - ".join([
            "{}: {:.3f}".format(k, v.item())
            for k, v in logging_vars.items()
        ])

        add_to_summary = self.elapsed_iterations % self.summary_interval == 0

        if self.is_main_process:
            self.logger.add_training_point(self.elapsed_iterations, add_to_summary, **logging_vars)

        dist_utils.synchronize()

        if self.is_main_process:
            if self.elapsed_iterations % self.display_interval == 0:
                print_fn = self.console_logger.info
            else:
                print_fn = self.console_logger.debug

            eta, avg_time_per_iter = self.logger.compute_eta(as_string=True)
            lr_string = self._get_learning_rate_log_string()

            print_str = "It: {:05d} - {:s} - lr: {} - ETA: {:s} - sec/it: {:.3f}".format(
                self.elapsed_iterations, logging_vars_str, lr_string, eta, avg_time_per_iter)

            print_fn(print_str)

        return

    def _get_learning_rate_log_string(self):
        lrs = []
        for opt, lr_schedulers in self.optimizers_and_lr_schedulers:
            if lr_schedulers is None:
                assert len(opt.param_groups) == 1
                lrs.append(opt.param_groups[0]['lr'])
            else:
                assert len(lr_schedulers[-1].get_last_lr()) == 1
                lrs.append(lr_schedulers[-1].get_last_lr()[0])

        lrs = ["{:.2E}".format(lr) for lr in lrs]

        if len(lrs) == 1:
            return lrs[0]
        else:
            return "[" + ", ".join(lrs) + "]"

    def get_gradient_clip_value(self) -> float:
        if self.elapsed_iterations < 100:
            return 20.0
        elif 50 <= self.elapsed_iterations < 1000:
            return 10.0
        else:
            return 2.0

    def get_oom_error_extra_details(self, training_sample: Any) -> str:
        return ""

    def get_pretraining_print_list(self) -> OrderedDict:
        return OrderedDict([])

    def get_image_summaries(self, training_sample: Any, model_output: Any) -> Dict[str, np.ndarray]:
        """
        The child class implementation of this method should return a dict mapping str to images as NumPy arrays
        :param training_sample:
        :param model_output:
        :return: Dict with keys of type str without white space and special characters. Values should be images
        as NumPay arrays of shape [B, C, H, W]. Dtype should either by uint8 with values in range [0, 255] or float32
        with values in range [0, 1]
        """
        return {}

    @staticmethod
    def is_oom_error(msg):
        return msg.startswith("RuntimeError('CUDA out of memory.")
