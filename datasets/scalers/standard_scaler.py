from typing import Tuple

import numpy as np
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
        super(StandardScaler).__init__()
        self.bias = offset
        self.scale = scale
        self.axis = axis

    def params(self):
        return dict(bias=self.bias, scale=self.scale)

    def fit(self, x: torch.Tensor, mask=None, keepdims: bool = True):
        if mask is not None:
            x_np = x.numpy()
            x_np = np.where(mask.numpy(), x_np, np.nan)
            self.bias = np.nanmean(x_np, axis=self.axis, keepdims=keepdims)
            self.scale = np.nanstd(x_np, axis=self.axis, keepdims=keepdims)
            self.bias = torch.from_numpy(self.bias)
            self.scale = torch.from_numpy(self.scale)
            print(
                f"{self.bias.min()=} {self.bias.max()=} {self.bias.mean()=} {self.bias.std()=}"
            )
            print(
                f"{self.scale.min()=} {self.scale.max()=} {self.scale.mean()=} {self.scale.std()=}"
            )
            bias = torch_nanmean(x, mask, axis=self.axis, keepdims=keepdims)
            scale = torch_nanstd(x, mask, axis=self.axis, keepdims=keepdims)
            print(f"{bias.min()=} {bias.max()=} {bias.mean()=} {bias.std()=}")
            print(f"{scale.min()=} {scale.max()=} {scale.mean()=} {scale.std()=}")

        else:
            self.bias = x.mean(axis=self.axis, keepdims=keepdims)
            self.scale = x.std(axis=self.axis, keepdims=keepdims)
