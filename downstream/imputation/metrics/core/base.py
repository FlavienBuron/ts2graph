import torch
import torch.nn.functional as F
from einops import reduce

epsilon = 1e-6


def mae(prediction: torch.Tensor, target: torch.Tensor, reduction="none"):
    return F.l1_loss(prediction, target, reduction=reduction)


def mse(prediction: torch.Tensor, target: torch.Tensor, reduction="none"):
    return F.mse_loss(prediction, target, reduction=reduction)


def mre(prediction: torch.Tensor, target: torch.Tensor, reduction="none"):
    return torch.abs(prediction - target) / (torch.abs(target) + epsilon)


def mape(prediction: torch.Tensor, target: torch.Tensor):
    return torch.abs((prediction - target) / target)


def wape(prediction: torch.Tensor, target: torch.Tensor):
    abs_diff = torch.abs(prediction - target)
    return abs_diff.sum() / (target.sum() + epsilon)


def smape(prediction: torch.Tensor, target: torch.Tensor):
    valid_mask = torch.abs(target) > epsilon
    abs_diff = torch.abs(prediction - target)
    abs_sum = torch.abs(prediction + target) + epsilon
    loss = 2 * abs_diff / abs_sum * valid_mask.float()
    return loss.sum() / valid_mask.sum()


def peak_prediction_loss(
    prediction: torch.Tensor, target: torch.Tensor, reduction: str = "none"
):
    true_obs_max = reduce(target, "b s n 1 -> b 1 n 1", "max")
    true_obs_min = reduce(target, "b s n 1 -> b 1 n 1", "min")
    target = torch.cat([true_obs_max, true_obs_min], dim=1)
    return F.mse_loss(prediction, target, reduction=reduction)


def wrap_loss_fn(base_loss):
    pass
