from typing import Optional
from torch import Tensor
import torch


def multiclass_dice_loss(input: Tensor, target: Tensor, eps: float = 1e-6,
                         check_target_validity: bool = True, ignore_zero_class: bool = True,
                         ignore_mask: Optional[Tensor] = None) -> Tensor:
    """
    Computes DICE loss for multi-class predictions. API inputs are identical to torch.nn.functional.cross_entropy()
    :param input: tensor of shape [N, C, *] with unscaled logits
    :param target: tensor of shape [N, *]
    :param eps:
    :param check_target_validity: checks if the values in the target are valid
    :param ignore_zero_class: Ignore the IoU for class ID 0
    :param ignore_mask: optional tensor of shape [N, *]
    :return: tensor
    """
    assert input.ndim >= 2
    input = input.softmax(1)
    num_classes = input.size(1)

    if check_target_validity:
        class_ids = target.unique()
        assert not torch.any(torch.logical_or(class_ids < 0, class_ids >= num_classes)), \
            f"Number of classes = {num_classes}, but target has the following class IDs: {class_ids.tolist()}"

    target = torch.stack([target == cls_id for cls_id in range(0, num_classes)], 1).to(dtype=input.dtype)  # [N, C, *]

    if ignore_zero_class:
        input = input[:, 1:]
        target = target[:, 1:]

    if ignore_mask is not None:
        ignore_mask = ignore_mask.unsqueeze(1)
        expand_dims = [-1, input.size(1)] + ([-1] * (ignore_mask.ndim - 2))
        ignore_mask = ignore_mask.expand(*expand_dims)

    return dice_loss(input, target, eps=eps, ignore_mask=ignore_mask)


def dice_loss_with_logits(input: Tensor, target: Tensor, ignore_mask: Optional[Tensor] = None,
                          eps: Optional[float] = 1e-6):
    return dice_loss(input.sigmoid(), target, ignore_mask, eps)


def dice_loss(input: Tensor, target: Tensor, ignore_mask: Optional[Tensor] = None, eps: Optional[float] = 1e-6):
    """
    Computes the DICE or soft IoU loss.
    :param input: tensor of shape [N, *]
    :param target: tensor with shape identical to input
    :param ignore_mask: tensor of same shape as input. non-zero values in this mask will be
    :param eps
    excluded from the loss calculation.
    :return: tensor
    """
    assert input.shape == target.shape, f"Shape mismatch between input ({input.shape}) and target ({target.shape})"
    assert input.dtype == target.dtype

    if torch.is_tensor(ignore_mask):
        assert ignore_mask.dtype == torch.bool
        assert input.shape == ignore_mask.shape, f"Shape mismatch between input ({input.shape}) and " \
            f"ignore mask ({ignore_mask.shape})"
        input = torch.where(ignore_mask, torch.zeros_like(input), input)
        target = torch.where(ignore_mask, torch.zeros_like(target), target)

    input = input.flatten(1)
    target = target.detach().flatten(1)

    numerator = 2.0 * (input * target).mean(1)
    denominator = (input + target).mean(1)

    soft_iou = (numerator + eps) / (denominator + eps)

    return torch.where(numerator > eps, 1. - soft_iou, soft_iou * 0.).mean()
