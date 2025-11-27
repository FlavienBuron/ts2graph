from typing import Tuple

import numpy as np

from datasets.scalers.abstract_scaler import AbstractScaler


class MinMaxScaler(AbstractScaler):
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

    def fit(self, x, mask=None, keepdims: bool = True):
        print(f"{type(x)=} {type(mask)=} {keepdims=}")
        if mask is not None:
            x = np.where(mask, x, np.nan)
            self.bias = np.nanmin(x, axis=self.axis, keepdims=keepdims)
            self.scale = np.nanmax(x, axis=self.axis, keepdims=keepdims) - self.bias
        else:
            self.bias = x.min(axis=self.axis, keepdims=keepdims)
            self.scale = x.max(axis=self.axis, keepdims=keepdims) - self.bias
