import torch
from torchmetrics import Metric


class Runtime(Metric):
    full_state_update = False

    def __init__(self, dist_sync_on_step=False, prefix=""):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # states for aggregation
        self.add_state("sum_time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_sq_time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.prefix = prefix  # optional prefix for MetricCollection names

    @torch.no_grad()
    def update(self, timing: float):
        t = torch.tensor(timing, device=self.sum_time.device)
        self.sum_time += t
        self.sum_sq_time += t * t
        self.count += 1

    def compute_total_time(self):
        return self.sum_time

    def compute_avg_time(self):
        mean = self.sum_time / self.count.clamp(min=1)
        return mean

    def compute_std_time(self):
        mean = self.sum_time / self.count.clamp(min=1)
        var = self.sum_sq_time / self.count.clamp(min=1) - mean**2
        std = torch.sqrt(torch.clamp(var, min=0))
        return std
