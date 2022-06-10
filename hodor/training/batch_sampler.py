from typing import Union
from torch.utils.data import Sampler
from torch.utils.data import Dataset

import math
import torch


class CustomBatchSampler(object):
    def __init__(self, sampler: Union[None, Sampler[int]], dataset: Dataset, batch_size: int,
                 post_shuffle: bool, max_allowed_ar_diff: float = 0.1, elapsed_batches: int = 0) -> None:

        if sampler is None:
            sample_idxes = torch.as_tensor(list(range(len(dataset))), dtype=torch.long)
        else:
            sample_idxes = torch.as_tensor(list(sampler), dtype=torch.long)

        assert len(sample_idxes) % batch_size == 0, f"Number of samples {len(sample_idxes)} must be exactly divisible " \
            f"by batch size ({batch_size})"

        if torch.is_tensor(dataset.sample_image_dims):
            sample_hw_dims = dataset.sample_image_dims.float()  # [num_samples, 2]    
        else:
            sample_hw_dims = torch.as_tensor(dataset.sample_image_dims, dtype=torch.float32)  # [num_samples, 2]

        sample_hw_dims = sample_hw_dims[sample_idxes]
        sample_ars = sample_hw_dims[:, 0] / sample_hw_dims[:, 1]

        if torch.is_tensor(dataset.sample_num_instances):
            sample_num_instances = dataset.sample_num_instances
        else:
            sample_num_instances = torch.as_tensor(dataset.sample_num_instances, dtype=torch.long)  # [num_samples]
            
        sample_num_instances = sample_num_instances[sample_idxes]

        ni_vals, ni_counts = sample_num_instances.unique(sorted=True, return_counts=True)
        # print(ni_vals.tolist(), ni_counts.tolist())

        batch_idxes = []
        total_padded_batches = 0

        for num_inst in ni_vals:
            indices = sample_num_instances == num_inst
            sample_idxes_num_inst = sample_idxes[indices]
            sample_ars_num_inst = sample_ars[indices]

            batch_idxes_num_inst, num_padded_batches = self.create_minibatches(
                sample_ars_num_inst, sample_idxes_num_inst, batch_size, max_allowed_ar_diff
            )

            batch_idxes.append(batch_idxes_num_inst)
            total_padded_batches += num_padded_batches

        batch_idxes = torch.cat(batch_idxes)
        # print(f"Total discarded batches: {total_padded_batches}")

        num_expected_batches = len(sample_idxes) // batch_size
        assert len(sample_idxes) % batch_size == 0

        if batch_idxes.size(0) < num_expected_batches:
            n_pad = num_expected_batches - batch_idxes.size(0)
            pad_idxes = torch.randint(0, batch_idxes.size(0) + 1, (n_pad,))
            batch_idxes = torch.cat((batch_idxes, batch_idxes[pad_idxes]))

        elif batch_idxes.size(0) > num_expected_batches:
            batch_idxes = batch_idxes[:num_expected_batches]

        if post_shuffle:
            randperm = torch.randperm(num_expected_batches)
            batch_idxes = batch_idxes[randperm]

        self.batch_idxes = batch_idxes

        if elapsed_batches > 0:
            assert elapsed_batches < len(self.batch_idxes)
            self.batch_idxes = self.batch_idxes[elapsed_batches:]

    @staticmethod
    def create_minibatches(sample_ars, sample_idxes, batch_size, max_allowed_ar_diff):
        n_pad = int((math.ceil(float(sample_ars.numel()) / batch_size) * batch_size) - sample_ars.numel())

        if n_pad > 0:
            sample_ars = torch.cat((sample_ars, sample_ars[:n_pad]))
            sample_idxes = torch.cat((sample_idxes, sample_idxes[:n_pad]))

        sample_ars, sorted_indices = sample_ars.sort()
        sorted_indices = sample_idxes[sorted_indices]

        sample_ars = sample_ars.reshape(-1, batch_size)  # [num_batches, batch_size]
        sorted_indices = sorted_indices.reshape(-1, batch_size)

        ar_diffs = sample_ars.max(dim=1)[0] - sample_ars.min(dim=1)[0]

        valid_indices = ar_diffs <= max_allowed_ar_diff

        num_invalid_batches = sample_ars.size(0) - valid_indices.sum(dtype=torch.long).item()
        # print("Num replacements: {}/{}".format(self.num_invalid_batches, valid_indices.size(0)))

        sorted_indices = sorted_indices[valid_indices]

        return sorted_indices, num_invalid_batches

    def __iter__(self):
        for i in range(len(self)):
            yield self.batch_idxes[i].tolist()

    def __len__(self):
        return len(self.batch_idxes)
