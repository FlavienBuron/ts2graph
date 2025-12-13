from typing import Tuple

import numpy as np

from datasets.scalers.abstract_scaler import AbstractScaler


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
        super(StandardScaler, self).__init__()

    def params(self):
        return dict(bias=self.bias, scale=self.scale)

    def fit(self, x, mask=None, keepdims: bool = True):
        if mask is not None:
            x = np.where(mask, x, np.nan)
            self.bias = np.nanmean(x, axis=self.axis, keepdims=keepdims)
            self.scale = np.nanstd(x, axis=self.axis, keepdims=keepdims)
        else:
            self.bias = x.mean(axis=self.axis, keepdims=keepdims)
            self.scale = x.std(axis=self.axis, keepdims=keepdims)
