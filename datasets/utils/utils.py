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
    if mask is None:
        mask = torch.isnan(x)

    if axis is None:
        result, _ = torch.min(x.masked_fill(mask, float("inf")))
        return result

    if isinstance(axis, int):
        axis = (axis,)

    current_x = x.clone()
    for dim in tuple(sorted(axis, reverse=True)):
        current_x, _ = torch.min(
            current_x.masked_fill(mask, float("inf")), dim=dim, keepdim=keepdims
        )

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
    if mask is None:
        mask = torch.isnan(x)

    if axis is None:
        result, _ = torch.max(x.masked_fill(mask, float("-inf")))
        return result

    if isinstance(axis, int):
        axis = (axis,)

    current_x = x.clone()
    for dim in tuple(sorted(axis, reverse=True)):
        current_x, _ = torch.max(
            current_x.masked_fill(mask, float("-inf")), dim=dim, keepdim=keepdims
        )

    return x
