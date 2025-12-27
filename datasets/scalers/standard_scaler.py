from typing import Tuple

import torch

from datasets.scalers.abstract_scaler import AbstractScaler
from datasets.utils.utils import torch_nanmean, torch_nanstd


class StandardScaler(AbstractScaler):
    def __init__(
        self,
        offset: float = 0.0,
        scale: float = 1.0,
        axis: Tuple = (0,),
    ) -> None:
        self.bias = offset
        self.scale = scale
        self.axis = axis

    def params(self):
        return dict(bias=self.bias, scale=self.scale)

    def fit(self, x: torch.Tensor, mask=None, keepdims: bool = True):
        if mask is not None:
            self.bias = torch_nanmean(x, mask, axis=self.axis, keepdims=keepdims)
            self.scale = torch_nanstd(x, mask, axis=self.axis, keepdims=keepdims)
        else:
            self.bias = x.mean(axis=self.axis, keepdims=keepdims)
            self.scale = x.std(axis=self.axis, keepdims=keepdims)
