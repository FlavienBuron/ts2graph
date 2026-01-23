import torch
from torchmetrics import Metric


class TotalRuntime(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__(dist_sync_on_step=False)

        self.add_state("sum_time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_sq_time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, timing: float):
        t = torch.tensor(timing, device=self.sum_time.device)
        self.sum_time += t
        self.sum_sq_time += t * t
        self.count += 1

    def compute(self):
        return self.sum_time


class MeanRuntime(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__(dist_sync_on_step=False)

        self.add_state("sum_time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_sq_time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, timing: float):
        t = torch.tensor(timing, device=self.sum_time.device)
        self.sum_time += t
        self.sum_sq_time += t * t
        self.count += 1

    def compute(self):
        mean = self.sum_time / self.count.clamp(min=1)

        return mean


class StdRuntime(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__(dist_sync_on_step=False)

        self.add_state("sum_time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_sq_time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, timing: float):
        t = torch.tensor(timing, device=self.sum_time.device)
        self.sum_time += t
        self.sum_sq_time += t * t
        self.count += 1

    def compute(self):
        mean = self.sum_time / self.count.clamp(min=1)
        var = self.sum_sq_time / self.count.clamp(min=1) - mean**2
        std = torch.sqrt(torch.clamp(var, min=0))

        return std
