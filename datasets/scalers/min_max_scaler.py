from typing import Tuple

import numpy as np
import torch

from datasets.scalers.abstract_scaler import AbstractScaler


class MinMaxScaler(AbstractScaler):
    def __init__(
        self,
        offset: float = 0.0,
        scale: float = 1.0,
        axis: Tuple = (0,),
    ) -> None:
        super(MinMaxScaler).__init__()
        self.bias = offset
        self.scale = scale
        self.axis = axis

    def params(self):
        return dict(bias=self.bias, scale=self.scale)

    def fit(self, x: torch.Tensor, mask=None, keepdims: bool = True):
        x_np = x.numpy()
        if mask is not None:
            x_np = np.where(mask.numpy(), x_np, np.nan)
            self.bias = np.nanmin(x_np, axis=self.axis, keepdims=keepdims)
            self.scale = np.nanmax(x_np, axis=self.axis, keepdims=keepdims)
            self.bias = torch.from_numpy(self.bias)
            self.scale = torch.from_numpy(self.scale)
        else:
            self.bias = x_np.min(axis=self.axis, keepdims=keepdims)
            self.scale = x_np.max(axis=self.axis, keepdims=keepdims) - self.bias
