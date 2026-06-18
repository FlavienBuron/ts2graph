from typing import Optional

import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("twed")
class TWED(DistanceFunction):
    name = "twed"
    input_kind = "series"
    symmetric = True
    non_negative = True
    supports_mask = True
    bounded = False

    def __init__(
        self,
        lambda_: float = 1.0,
        nu: float = 0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_ = lambda_
        self.nu = nu

    def __call__(self, X: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:

        T, N, F = X.shape

        D = torch.full((N, N), float("inf"), device=X.device, dtype=X.dtype)

        time = torch.arange(T, device=X.device, dtype=X.dtype)

        for i in range(N):
            Xi = X[:, i, :].clone()

            for j in range(i + 1, N):
                Xj = X[:, j, :].clone()

                if mask is not None:
                    Mi = mask[:, i, :].bool()
                    Mj = mask[:, j, :].bool()

                    Xi = Xi.clone()
                    Xj = Xj.clone()

                    Xi[~Mi] = 0.0
                    Xj[~Mj] = 0.0

                if F == 1:
                    Xi = Xi.squeeze(-1)
                    Xj = Xj.squeeze(-1)

                n, m = Xi.shape[0], Xj.shape[0]

                costs = torch.full(
                    (n + 1, m + 1),
                    float("inf"),
                    device=X.device,
                    dtype=X.dtype,
                )

                costs[0, 0] = 0.0

                # initialization
                for p in range(1, n + 1):
                    costs[p, 0] = costs[p - 1, 0] + torch.linalg.norm(Xi[p - 1] - Xi[max(p - 2, 0)]) + self.nu * (time[p - 1] - time[max(p - 2, 0)]) + self.lambda_

                for q in range(1, m + 1):
                    costs[0, q] = costs[0, q - 1] + torch.linalg.norm(Xj[q - 1] - Xj[max(q - 2, 0)]) + self.nu * (time[q - 1] - time[max(q - 2, 0)]) + self.lambda_

                # recursion
                for p in range(1, n + 1):
                    for q in range(1, m + 1):
                        dist_xy = torch.linalg.norm(Xi[p - 1] - Xj[q - 1])
                        dist_xx = torch.linalg.norm(Xi[p - 1] - Xi[p - 2])
                        dist_yy = torch.linalg.norm(Xj[q - 1] - Xj[q - 2])

                        match = costs[p - 1, q - 1] + dist_xy + self.nu * (time[p - 1] - time[q - 1]).abs()

                        delete_x = costs[p - 1, q] + dist_xx + self.nu * (time[p - 1] - time[p - 2]).abs() + self.lambda_

                        delete_y = costs[p, q - 1] + dist_yy + self.nu * (time[q - 1] - time[q - 2]).abs() + self.lambda_

                        costs[p, q] = (torch.stack([match, delete_x, delete_y])).min()

                D[i, j] = D[j, i] = costs[n, m]

        D.fill_diagonal_(0.0)

        return D
