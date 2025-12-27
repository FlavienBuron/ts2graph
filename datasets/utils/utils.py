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


def torch_nanmean(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    axis: int | Tuple | None = None,
    keepdims=False,
    eps: float = 1e-8,
):
    """
    Replicate the behavior of Numpy's `nanmean`.
    True in mask = missing value (ignore)
    """
    # Create mask: True means missing/ignore, False means keep
    if mask is None:
        mask = torch.isnan(x)

    # Invert mask for easier logic: True = keep, False = ignore
    keep_mask = ~mask

    # Fill missing values with 0 for sum
    x_filled = x.masked_fill(mask, 0.0)

    if axis is None:
        # Compute sum and count over all elements
        total = x_filled.sum()
        count = keep_mask.sum().clamp_min(eps)
        result = total / count
        return result if keepdims else result

    # Handle single axis as tuple
    if isinstance(axis, int):
        axis = (axis,)

    # Compute sum over specified axes
    total = x_filled.sum(dim=axis, keepdim=keepdims)
    count = keep_mask.sum(dim=axis, keepdim=keepdims).clamp_min(eps)

    return total / count


def torch_nanstd(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    axis: int | Tuple | None = None,
    keepdims=False,
    unbiased: bool = True,
    eps: float = 1e-8,
):
    """
    Replicate the behavior of Numpy's `nanstd`.
    True in mask = missing value (ignore)
    """
    # Create mask: True means missing/ignore, False means keep
    if mask is None:
        mask = torch.isnan(x)

    # Invert mask for easier logic: True = keep, False = ignore
    keep_mask = ~mask

    # First compute the mean using valid values only
    mean = torch_nanmean(x, mask=mask, axis=axis, keepdims=True, eps=eps)

    if axis is None:
        # For full reduction
        # Compute squared differences only for valid values
        sq_diff = (x - mean) ** 2
        sq_diff_filled = sq_diff.masked_fill(mask, 0.0)
        total = sq_diff_filled.sum()
        count = keep_mask.sum()

    else:
        # For partial reduction
        if isinstance(axis, int):
            axis = (axis,)

        # Expand mean for broadcasting
        mean_expanded = mean
        for dim in axis:
            if keepdims:
                mean_expanded = mean_expanded.unsqueeze(dim)

        # Ensure proper broadcasting
        while mean_expanded.dim() < x.dim():
            mean_expanded = mean_expanded.unsqueeze(-1)

        # Compute squared differences only for valid values
        sq_diff = (x - mean_expanded) ** 2
        sq_diff_filled = sq_diff.masked_fill(mask, 0.0)
        total = sq_diff_filled.sum(dim=axis, keepdim=keepdims)
        count = keep_mask.sum(dim=axis, keepdim=keepdims)

    # Apply Bessel's correction if unbiased
    if unbiased:
        count = count - 1

    # Avoid division by zero
    count = count.clamp_min(eps)

    variance = total / count
    return torch.sqrt(variance)
