from typing import Tuple

import numpy as np
import torch

from datasets.scalers.abstract_scaler import AbstractScaler
from datasets.utils.utils import torch_nanmax, torch_nanmin


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
        if mask is not None:
            print(f"{type(mask)=} {type(x)=}")
            x = torch.where(mask, np.nan, x)

            self.bias = torch_nanmin(x, mask, axis=self.axis, keepdims=keepdims)
            self.scale = (
                torch_nanmax(x, mask, axis=self.axis, keepdims=keepdims) - self.bias
            )
        else:
            self.bias = x.min(axis=self.axis, keepdims=keepdims)
            self.scale = x.max(axis=self.axis, keepdims=keepdims) - self.bias
