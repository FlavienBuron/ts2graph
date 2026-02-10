from typing import Optional

import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("identity")
class Identity(DistanceFunction):
    input_kind = "coordinates"

    def __init__(self, input: str, **kwargs):
        super().__init__(**kwargs)
        self.input_kind = input

    def __call__(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return x.detach().clone()
