import torch


class RuntimeAccumulator:
    """Accumulate batch timings and compute total, mean, std per epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._timings = []

    @torch.no_grad()
    def update(self, timing: float):
        self._timings.append(float(timing))

    def compute(self):
        if not self._timings:
            t = torch.tensor([0.0])
        else:
            t = torch.tensor(self._timings)

        return {"total_time": t.sum(), "avg_time": t.mean(), "std_time": t.std()}
