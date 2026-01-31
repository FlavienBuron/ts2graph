import torch
from torchmetrics import Metric


class MaskedCorrelation(Metric):
    full_state_update = False

    def __init__(self, at=None):
        super().__init__()
        self.at = slice(None) if at is None else slice(at, at + 1)

        # generic counters
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def _select(self, prediction, target, mask):
        prediction = prediction[:, self.at]
        target = target[:, self.at]
        mask = mask[:, self.at].bool()

        x = prediction[mask]
        y = target[mask]
        return x, y

    def update(self, prediction, target, mask):
        x, y = self._select(prediction, target, mask)

        if x.numel() == 0:
            return

        self._update_stats(x, y)
        self.n += x.numel()

    def _update_stats(self, x: torch.Tensor, y: torch.Tensor):
        """
        Implement in subclass.
        Update sufficient statistics using masked x, y.
        """
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        if self.n == 0:
            return torch.tensor(0.0, device=self.n.device)
        return self._compute_corr()

    def _compute_corr(self) -> torch.Tensor:
        """
        Implement in subclass.
        Return a scalar tensor.
        """
        raise NotImplementedError
