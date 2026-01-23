import torch


class Runtime:
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
            timing = torch.tensor([0.0])
        else:
            timing = torch.tensor(self._timings)

        return {
            "total_time": timing.sum(),
            "avg_time": timing.mean(),
            "std_time": timing.std(),
        }
