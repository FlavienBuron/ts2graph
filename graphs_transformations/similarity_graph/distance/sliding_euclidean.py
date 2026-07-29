from typing import Optional

import torch
import torch.fft

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("sliding_euclidean")
class SlidingEuclidean(DistanceFunction):
    name = "sliding euclidean"
    input_kind = "series"
    symmetric = True
    non_negative = True
    supports_mask = True
    bounded = False

    def __init__(
        self,
        lag_fraction: float = 1.0,
        min_overlap: int = 1,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.lag_fraction = lag_fraction
        self.min_overlap = min_overlap
        self.normalize = normalize

    def _max_lag(self, T: int) -> int:
        """
        Interpret lag_fraction:

        - None       -> all lags
        - 0.0        -> only lag 0
        - 0.2        -> +/- 20% of T-1
        - 1.0        -> +/- (T-1)
        - > 1.0      -> treat as absolute number of lags
        """
        if self.lag_fraction is None:
            return T - 1

        lf = float(self.lag_fraction)

        if lf <= 0:
            return 0

        if lf <= 1.0:
            return int(round(lf * (T - 1)))

        return int(min(lf, T - 1))

    def fft_cross_correlation(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Full linear cross-correlation for 1D tensors.

        Output lags are ordered:

            [-(n-1), ..., -1, 0, 1, ..., n-1]

        The zero-lag entry is therefore at index n-1.
        """
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("fft_cross_correlation expects 1D tensors")

        a = a.float()
        b = b.float()

        n = a.shape[0]

        if n == 1:
            return (a * b).sum().view(1)

        # Use enough points for linear correlation.
        # Power-of-two is usually faster.
        full_len = 2 * n - 1
        nfft = 1 << (full_len - 1).bit_length()

        A = torch.fft.rfft(a, n=nfft)
        B = torch.fft.rfft(b, n=nfft)

        corr = torch.fft.irfft(A * torch.conj(B), n=nfft)

        # Negative lags are stored at the end.
        # Nonnegative lags are stored at the beginning.
        return torch.cat([corr[-(n - 1) :], corr[:n]])

    def __call__(self, X: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        T, N, F = X.shape

        if F != 1:
            raise ValueError("SlidingEuclidean currently expects univariate series with F=1.")

        X = X.squeeze(-1).float()

        if mask is None:
            mask = torch.ones_like(X, dtype=X.dtype)
        else:
            mask = mask.squeeze(-1).float()

        if X.ndim != 2:
            raise ValueError(f"Expected X shape (T, N, 1), got {tuple(X.shape)}")

        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape (T, N, 1), got {tuple(mask.shape)}")

        if X.shape != mask.shape:
            raise ValueError(f"X and mask must have the same shape after squeezing. Got X={tuple(X.shape)}, mask={tuple(mask.shape)}")

        D = torch.full((N, N), float("inf"), device=X.device, dtype=X.dtype)

        max_lag = self._max_lag(T)
        center = T - 1

        for i in range(N):
            xi = X[:, i]
            mi = mask[:, i]

            xi_m = xi * mi
            xi2_m = xi_m * xi_m

            for j in range(i + 1, N):
                xj = X[:, j]
                mj = mask[:, j]

                xj_m = xj * mj
                xj2_m = xj_m * xj_m

                # Lag-dependent terms.
                overlap = self.fft_cross_correlation(mi, mj)
                cross = self.fft_cross_correlation(xi_m, xj_m)

                # These are the missing lag-dependent squared terms.
                xi2_overlap = self.fft_cross_correlation(xi2_m, mj)
                xj2_overlap = self.fft_cross_correlation(mi, xj2_m)

                # Restrict to requested lag range.
                if max_lag < center:
                    lo = center - max_lag
                    hi = center + max_lag + 1

                    overlap = overlap[lo:hi]
                    cross = cross[lo:hi]
                    xi2_overlap = xi2_overlap[lo:hi]
                    xj2_overlap = xj2_overlap[lo:hi]

                # Numerical cleanup.
                # Overlap should be an integer count if masks are binary.
                overlap = overlap.clamp(min=0.0)

                # If masks are strictly binary, this can help with FFT noise:
                # overlap = torch.round(overlap)

                valid = overlap >= self.min_overlap

                if not bool(valid.any()):
                    # Leave D[i, j] as inf.
                    continue

                dist2 = xi2_overlap + xj2_overlap - 2.0 * cross

                # FFT roundoff can create tiny negative values.
                dist2 = dist2.clamp(min=0.0)

                if self.normalize:
                    # Safe division; invalid lags will be ignored anyway.
                    dist2 = dist2 / overlap.clamp(min=1.0)

                dist = torch.sqrt(dist2)

                best = dist[valid].min()

                D[i, j] = best
                D[j, i] = best

        D.fill_diagonal_(0.0)

        return D
