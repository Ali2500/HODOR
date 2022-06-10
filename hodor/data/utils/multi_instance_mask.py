from PIL import Image

import numpy as np
import torch


class MultiInstanceMask(object):
    def __init__(self, mask):
        if isinstance(mask, Image.Image):
            assert mask.mode == 'L'
            self._mask = mask
        elif isinstance(mask, np.ndarray):
            self._mask = Image.fromarray(mask.astype(np.uint8), mode='L')
        else:
            assert torch.is_tensor(mask)
            self._mask = Image.fromarray(mask.numpy().astype(np.uint8), mode='L')

    def __getitem__(self, index):
        return self._mask[index]

    def copy(self):
        return self.__class__(self._mask.clone())

    def numpy(self):
        return np.array(self._mask)

    def tensor(self):
        return torch.from_numpy(np.array(self._mask))

    def resize(self, size):
        h, w = size
        return self.__class__(self._mask.resize((w, h), resample=Image.NEAREST))

    def flip_horizontal(self):
        return self.__class__(self._mask.transpose(Image.FLIP_LEFT_RIGHT))

    height = property(fget=lambda self: self._mask.size[1])
    width = property(fget=lambda self: self._mask.size[0])
    shape = property(fget=lambda self: self._mask.size[::-1])
