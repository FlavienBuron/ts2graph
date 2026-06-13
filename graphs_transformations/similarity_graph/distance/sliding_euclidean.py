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

    def __init__(self, lag_fraction: int, min_overlap: int = 1, normalize: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lag_fraction = lag_fraction
        self.min_overlap = min_overlap
        self.normalize = normalize

    def fft_cross_correlation(self, a, b):
        # full linear cross-correlation via FFT
        n = a.shape[0]
        size = 2 * n

        A = torch.fft.rfft(a, n=size)
        B = torch.fft.rfft(b, n=size)

        corr = torch.fft.irfft(A * torch.conj(B), n=size)

        # convert to valid lags [-n, n]
        return torch.cat([corr[-(n - 1) :], corr[:n]])

    def __call__(self, X: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        T, N, F = X.shape
        X = X.squeeze(-1) if F == 1 else X
        mask = mask.squeeze(-1).float() if mask is not None else torch.ones_like(X)

        D = torch.full((N, N), float("inf"), device=X.device)

        for i in range(N):
            xi = X[:, i]
            mi = mask[:, i]

            xi2 = xi * xi

            for j in range(i + 1, N):
                xj = X[:, j]
                mj = mask[:, j]

                xj2 = xj * xj

                # masked signals
                xi_m = xi * mi
                xj_m = xj * mj

                # cross term (all lags at once)
                cross = self.fft_cross_correlation(xi_m, xj_m)

                # self-energy terms
                xi_energy = xi2.sum()
                xj_energy = xj2.sum()

                # overlap per lag (approx via correlation of masks)
                overlap = self.fft_cross_correlation(mi, mj).clamp(min=1.0)

                # squared distance for each lag
                dist2 = xi_energy + xj_energy - 2 * cross

                # normalize if needed
                dist = torch.sqrt(torch.clamp(dist2 / overlap, min=0.0))

                D[i, j] = D[j, i] = torch.min(dist)

        D.fill_diagonal_(0.0)
        return D

    # def __call__(self, X: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    #     print(f"Distance: {self.name}")
    #     if mask is None:
    #         raise ValueError("Masked Euclidean Distance requires a mask")
    #
    #     T, N, Fdim = X.shape
    #     device = X.device
    #     dtype = X.dtype
    #
    #     max_lag = int(round(self.lag_fraction * T))
    #     lags = torch.arange(-max_lag, max_lag + 1, device=device)
    #
    #     X = X.squeeze(-1) if Fdim == 1 else X
    #     mask = mask.squeeze(-1).bool()
    #
    #     D = torch.full((N, N), float("inf"), device=device, dtype=dtype)
    #
    #     min_overlap = self.min_overlap
    #     normalize = self.normalize
    #
    #     for i in range(N):
    #         Xi = X[:, i]
    #         Mi = mask[:, i]
    #
    #         # ---------------------------------------------------------
    #         # precompute all shifted Xi, Mi ONCE per i (vectorized)
    #         # ---------------------------------------------------------
    #         Xi_shift = []
    #         Mi_shift = []
    #
    #         for tau in lags:
    #             if tau >= 0:
    #                 x = Xi[: T - tau]
    #                 m = Mi[: T - tau]
    #             else:
    #                 lag = -tau
    #                 x = Xi[lag:]
    #                 m = Mi[lag:]
    #
    #             Xi_shift.append(x)
    #             Mi_shift.append(m)
    #
    #         max_len = max(x.shape[0] for x in Xi_shift)
    #
    #         def pad(x, val=0.0):
    #             if x.shape[0] == max_len:
    #                 return x
    #             return F.pad(x, (0, max_len - x.shape[0]), value=val)
    #
    #         Xi_shift = torch.stack([pad(x) for x in Xi_shift])  # [L, T]
    #         Mi_shift = torch.stack([pad(x.float()) for x in Mi_shift]).bool()
    #
    #         for j in range(i + 1, N):
    #             Xj = X[:, j]
    #             Mj = mask[:, j]
    #
    #             # ---------------------------------------------------------
    #             # vectorized shifts for j (NO tau loop per pair)
    #             # ---------------------------------------------------------
    #             Xj_shift = []
    #             Mj_shift = []
    #
    #             for tau in lags:
    #                 if tau >= 0:
    #                     x = Xj[: T - tau]
    #                     m = Mj[: T - tau]
    #                 else:
    #                     lag = -tau
    #                     x = Xj[lag:]
    #                     m = Mj[lag:]
    #
    #                 Xj_shift.append(x)
    #                 Mj_shift.append(m)
    #
    #             Xj_shift = torch.stack([pad(x) for x in Xj_shift])
    #             Mj_shift = torch.stack([pad(x.float()) for x in Mj_shift]).bool()
    #
    #             # ---------------------------------------------------------
    #             # fully vectorized over τ
    #             # ---------------------------------------------------------
    #             Mij = Mi_shift & Mj_shift
    #
    #             diff = Xi_shift - Xj_shift
    #             diff = diff * Mij
    #
    #             square = (diff * diff).sum(dim=1)  # [L]
    #             K = Mij.sum(dim=1).clamp(min=1)
    #
    #             valid = K >= min_overlap
    #
    #             if normalize:
    #                 D_tau = (square / K).sqrt()
    #             else:
    #                 D_tau = (square).sqrt()
    #
    #             D_tau = torch.where(valid, D_tau, torch.tensor(float("inf"), device=device, dtype=dtype))
    #
    #             best = torch.min(D_tau)
    #
    #             D[i, j] = D[j, i] = best
    #
    #     D.fill_diagonal_(0.0)
    #     return D
