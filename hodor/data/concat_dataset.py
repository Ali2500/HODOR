from torch.utils.data import Dataset, ConcatDataset as ConcatDatasetBase
from typing import List, Iterable


class ConcatDataset(ConcatDatasetBase):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__(datasets)
        self.sample_image_dims = sum([ds.sample_image_dims for ds in datasets], [])
        self.sample_num_instances = sum([ds.sample_num_instances for ds in datasets], [])
