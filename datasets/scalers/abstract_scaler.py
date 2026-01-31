from abc import ABC, abstractmethod

import numpy as np
import torch


class AbstractScaler(ABC):
    def __init__(
        self,
        **kwargs,
    ):
        self.offset = 0.0
        self.scale = 1.0
        self.axis = (0,)
        self.bias = 0.0
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def fit(self, x, mask=None, keepdims: bool = True):
        pass

    def transform(self, x):
        return (x - self.bias) / self.scale

    def inverse_transform(self, x):
        return x * self.scale + self.bias

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def params(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if not callable(v) and not k.startswith("__")
        }

    def to_torch(self):
        for p in self.params():
            param = getattr(self, p)
            param = np.atleast_1d(param)
            param = torch.tensor(param).float()
            setattr(self, p, param)
        return self
