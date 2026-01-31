import torch

from downstream.imputation.metrics.core.base import epsilon
from downstream.imputation.metrics.core.masked_correlation import MaskedCorrelation


class MaskedPearson(MaskedCorrelation):
    def __init__(self, at=None):
        super().__init__(at=at)

        self.add_state("sum_x", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_y", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_x2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_y2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_xy", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _update_stats(self, x, y):
        self.sum_x += x.sum()
        self.sum_y += y.sum()
        self.sum_x2 += (x**2).sum()
        self.sum_y2 += (y**2).sum()
        self.sum_xy += (x * y).sum()

    def _compute_corr(self):
        n = self.n.clamp(min=1)

        mean_x = self.sum_x / n
        mean_y = self.sum_y / n

        cov = self.sum_xy / n - mean_x * mean_y
        var_x = self.sum_x2 / n - mean_x**2
        var_y = self.sum_y2 / n - mean_y**2

        denom = torch.sqrt(var_x * var_y).clamp(min=epsilon)
        return cov / denom


class MaskedCCC(MaskedCorrelation):
    def __init__(self, at=None):
        super().__init__(at)

        self.add_state("sum_x", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_y", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_x2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_y2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_xy", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _update_stats(self, x, y):
        self.sum_x += x.sum()
        self.sum_y += y.sum()
        self.sum_x2 += (x**2).sum()
        self.sum_y2 += (y**2).sum()
        self.sum_xy += (x * y).sum()

    def _compute_corr(self):
        eps = 1e-8
        n = self.n.clamp(min=1)

        mean_x = self.sum_x / n
        mean_y = self.sum_y / n

        var_x = self.sum_x2 / n - mean_x**2
        var_y = self.sum_y2 / n - mean_y**2
        cov = self.sum_xy / n - mean_x * mean_y

        return (2 * cov) / (var_x + var_y + (mean_x - mean_y) ** 2 + eps)


class MaskedCosineSimilarity(MaskedCorrelation):
    def __init__(self, at=None):
        super().__init__(at)

        self.add_state("dot", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("norm_x", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("norm_y", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _update_stats(self, x, y):
        self.dot += (x * y).sum()
        self.norm_x += (x**2).sum()
        self.norm_y += (y**2).sum()

    def _compute_corr(self):
        eps = 1e-8
        return self.dot / (torch.sqrt(self.norm_x) * torch.sqrt(self.norm_y) + eps)


class MaskedLagCorrelation(MaskedCorrelation):
    def __init__(self, lag: int = 1, at=None):
        super().__init__(at)
        self.lag = lag

        self.add_state("sum_x", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_y", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_x2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_y2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_xy", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _select(self, prediction, target, mask):
        # prediction, target: [B, T, ..., F]
        prediction = prediction[..., self.at]
        target = target[..., self.at]
        mask = mask[..., self.at].bool()

        if prediction.size(1) <= self.lag:
            return None, None

        x = prediction[:, self.lag :]
        y = target[:, : -self.lag]
        m = mask[:, self.lag :] & mask[:, : -self.lag]

        if not m.any():
            return None, None

        return x[m], y[m]

    def update(self, prediction, target, mask):
        x, y = self._select(prediction, target, mask)
        if x is None:
            return

        self._update_stats(x, y)
        self.n += x.numel()

    def _update_stats(self, x, y):
        self.sum_x += x.sum()
        self.sum_y += y.sum()
        self.sum_x2 += (x**2).sum()
        self.sum_y2 += (y**2).sum()
        self.sum_xy += (x * y).sum()

    def _compute_corr(self):
        eps = 1e-8
        n = self.n.clamp(min=1)

        mean_x = self.sum_x / n
        mean_y = self.sum_y / n

        cov = self.sum_xy / n - mean_x * mean_y
        var_x = self.sum_x2 / n - mean_x**2
        var_y = self.sum_y2 / n - mean_y**2

        return cov / (torch.sqrt(var_x * var_y) + eps)
