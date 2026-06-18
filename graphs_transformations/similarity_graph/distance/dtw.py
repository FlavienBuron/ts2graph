from typing import Optional

import dtw_missing.dtw_missing as dtw_m
import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("dtw")
class DTW(DistanceFunction):
    name = "dtw"
    input_kind = "series"
    symmetric = True
    non_negative = True
    supports_mask = True
    bounded = False

    def __init__(
        self,
        series_fraction: float = 0.1,
        missing_value_restrictions: str = "full",
        missing_value_adjustment: str = "proportion_of_missing_values",
        use_c: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.series_fraction = series_fraction
        self.missing_value_restrictions = missing_value_restrictions
        self.missing_value_adjustment = missing_value_adjustment
        self.use_c = use_c

    def __call__(self, X: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:

        T, N, F = X.shape
        window_len = max(1, int(self.series_fraction * T))

        D = torch.full(
            (N, N),
            float("inf"),
        )

        for i in range(N):
            Xi = X[:, i, :].clone()

            for j in range(i + 1, N):
                Xj = X[:, j, :].clone()

                if mask is not None:
                    Mi = mask[:, i, :].bool()
                    Mj = mask[:, j, :].bool()

                    Xi = Xi.clone()
                    Xj = Xj.clone()

                    Xi[~Mi] = float("nan")
                    Xj[~Mj] = float("nan")

                xi_np = Xi.cpu().numpy()
                xj_np = Xj.cpu().numpy()

                if F == 1:
                    xi_np = xi_np.squeeze(-1)
                    xj_np = xj_np.squeeze(-1)

                cost = dtw_m.warping_paths(
                    s1=xi_np,
                    s2=xj_np,
                    window=window_len,
                    missing_value_restrictions=self.missing_value_restrictions,
                    missing_value_adjustment=self.missing_value_adjustment,
                    use_c=self.use_c,
                )[0]

                D[i, j] = D[j, i] = cost

        D.fill_diagonal_(0.0)

        return D
