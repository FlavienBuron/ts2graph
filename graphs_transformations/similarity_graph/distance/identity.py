from typing import Optional

import numpy as np
import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("identity")
class Identity(DistanceFunction):
    input_kind = "coordinates"

    def __init__(self, input: str, **kwargs):
        super().__init__(**kwargs)
        self.input_kind = input

    def __call__(
        self, x: torch.Tensor | np.ndarray, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        return x.detach().clone()
