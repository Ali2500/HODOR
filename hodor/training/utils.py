"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""
import torch
import torch.distributed as dist


# -------------------------------------------------- Distributed Training ----------------------------------------------
def is_distributed():
    if not dist.is_available():
        return False
    return dist.is_initialized()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def _get_device(x):
    if torch.is_tensor(x):
        assert x.device.type != "cpu"
        device = x.device
    elif isinstance(x, (str, torch.device)):
        device = x
    else:
        raise ValueError("Argument should be torch.Tensor, torch.device or str")

    return device


# ----------------------------------------------- CUDA memory profiling ------------------------------------------------
def _convert_unit(x, unit):
    valid_units = ("B", "KB", "MB", "GB")
    assert unit in valid_units, "Invalid unit '{}'".format(unit)
    return x / float(1024. ** valid_units.index(unit))


def get_allocated_memory(x, unit="B"):
    device = _get_device(x)
    allocated = torch.cuda.memory_allocated(device)
    return _convert_unit(allocated, unit)


def get_cached_memory(x, unit="B"):
    device = _get_device(x)
    cached = torch.cuda.memory_reserved(device)
    return _convert_unit(cached, unit)
