from typing import Tuple

import torch


def torch_nanmin(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    axis: int | Tuple | None = None,
    keepdims=False,
):
    """
    Replicate the behavior of Numpy's `nanmin`.
    """
    mask = torch.isnan(x) if mask is None else mask
    if axis is None:
        x, _ = torch.min(x.masked_fill(mask, float("inf")))
        return x
    if isinstance(axis, int):
        axis = (axis,)

    for dim in sorted(axis, reverse=True):
        x, _ = torch.min(x.masked_fill(mask, float("inf")), dim=dim, keepdim=keepdims)

    return x


def torch_nanmax(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    axis: int | Tuple | None = None,
    keepdims=False,
):
    """
    Replicate the behavior of Numpy's `nanmax`.
    """
    mask = torch.isnan(x) if mask is None else mask
    if axis is None:
        x, _ = torch.min(x.masked_fill(mask, float("-inf")))
        return x
    if isinstance(axis, int):
        axis = (axis,)

    for dim in sorted(axis, reverse=True):
        x, _ = torch.min(x.masked_fill(mask, float("inf")), dim=dim, keepdim=keepdims)

    return x
