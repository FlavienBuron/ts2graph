import hashlib
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple

import dtw_missing.dtw_missing as dtw_m
import h5py
import numpy as np
import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


def _compute_dtw_pair(args: Tuple) -> Tuple[int, int, float]:
    """
    Worker function to compute DTW for a single pair.
    Must be at the top level of the file to be picklable by multiprocessing.
    """
    i, j, xi_np, xj_np, mi_np, mj_np, window_len, restrictions, adjustment, use_c, F = args

    # 1. Inject NaNs based on masks
    if mi_np is not None:
        xi = xi_np.copy()
        xi[~mi_np] = np.nan
        xj = xj_np.copy()
        xj[~mj_np] = np.nan
    else:
        xi, xj = xi_np, xj_np

    # 2. Squeeze to 1D if univariate
    if F == 1:
        xi = xi.squeeze(-1)
        xj = xj.squeeze(-1)

    # 3. Compute DTW
    cost = dtw_m.warping_paths(
        s1=xi,
        s2=xj,
        window=window_len,
        missing_value_restrictions=restrictions,
        missing_value_adjustment=adjustment,
        use_c=use_c,
    )[0]

    return i, j, cost


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
        use_c: bool = False,
        cache_dir: Optional[str] = None,
        scenario_key: Optional[str] = None,
        n_jobs: int = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.series_fraction = series_fraction
        self.missing_value_restrictions = missing_value_restrictions
        self.missing_value_adjustment = missing_value_adjustment
        self.use_c = use_c
        self.cache_dir = cache_dir
        self.scenario_key = scenario_key
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()

        self._memory_cache = {}

    def __call__(self, X: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        print(f"Distance: {self.name}")
        print(f"[DEBUG] Using scenario key{self.scenario_key}")
        cache_key = self._get_cache_key()
        use_cache = cache_key is not None

        if use_cache:
            if cache_key in self._memory_cache:
                return self._memory_cache[cache_key]

            if self.cache_dir is not None:
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_path = os.path.join(self.cache_dir, f"dtw_{cache_key}.h5")

                if os.path.exists(cache_path):
                    try:
                        with h5py.File(cache_path, "r") as f:
                            D_np = f["distance_matrix"][:]

                        D = torch.from_numpy(D_np)
                        print(f"Loaded DTW pre-computed distance matrix found in cache: {cache_key}")
                        self._memory_cache[cache_key] = D
                        return D
                    except Exception as e:
                        print(f"Warning: Failed to read cache {cache_path}. Error: {e}")

        # T, N, F = X.shape
        # window_len = max(1, int(self.series_fraction * T))
        #
        # D = torch.full(
        #     (N, N),
        #     float("inf"),
        # )
        #
        # for i in range(N):
        #     Xi = X[:, i, :].clone()
        #
        #     for j in range(i + 1, N):
        #         Xj = X[:, j, :].clone()
        #
        #         if mask is not None:
        #             Mi = mask[:, i, :].bool()
        #             Mj = mask[:, j, :].bool()
        #
        #             Xi = Xi.clone()
        #             Xj = Xj.clone()
        #
        #             Xi[~Mi] = float("nan")
        #             Xj[~Mj] = float("nan")
        #
        #         xi_np = Xi.cpu().numpy()
        #         xj_np = Xj.cpu().numpy()
        #
        #         if F == 1:
        #             xi_np = xi_np.squeeze(-1)
        #             xj_np = xj_np.squeeze(-1)
        #
        #         cost = dtw_m.warping_paths(
        #             s1=xi_np,
        #             s2=xj_np,
        #             window=window_len,
        #             missing_value_restrictions=self.missing_value_restrictions,
        #             missing_value_adjustment=self.missing_value_adjustment,
        #             use_c=self.use_c,
        #         )[0]
        #
        #         D[i, j] = D[j, i] = cost

        T, N, F = X.shape
        window_len = max(1, int(self.series_fraction * T))

        # Convert to NumPy ONCE to avoid pickling overhead in multiprocessing
        X_np = np.ascontiguousarray(X.cpu().numpy())
        mask_np = np.ascontiguousarray(mask.cpu().numpy()) if mask is not None else None

        # Pre-extract 1D slices for each node.
        # This prevents passing the massive 3D array to every worker process.
        X_list = [X_np[:, i, :] for i in range(N)]
        mask_list = [mask_np[:, i, :] for i in range(N)] if mask_np is not None else [None] * N

        # Build the task list
        tasks = []
        for i in range(N):
            for j in range(i + 1, N):
                tasks.append(
                    (
                        i,
                        j,
                        X_list[i],
                        X_list[j],
                        mask_list[i],
                        mask_list[j],
                        window_len,
                        self.missing_value_restrictions,
                        self.missing_value_adjustment,
                        self.use_c,
                        F,
                    )
                )
        print(f"Computing {len(tasks):,} pairs using {self.n_jobs} CPU cores...")

        D = torch.full((N, N), float("inf"))

        # Use ProcessPoolExecutor to bypass the Python GIL
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # executor.map preserves order, but we just need to unpack the results
            for i, j, cost in executor.map(_compute_dtw_pair, tasks):
                D[i, j] = cost
                D[j, i] = cost

        D.fill_diagonal_(0.0)

        D.fill_diagonal_(0.0)

        if use_cache:
            self._memory_cache[cache_key] = D

            if self.cache_dir is not None:
                cache_path = os.path.join(self.cache_dir, f"dtw_{cache_key}.h5")
                temp_path = cache_path + ".tmp"

                with h5py.File(temp_path, "w") as f:
                    f.create_dataset("distance_matrix", data=D.cpu().numpy(), compression="gzip", compression_opts=4)
                os.replace(temp_path, cache_path)
                print(f"Cached DTW distance matrix {cache_key}")

        return D

    def _get_cache_key(self) -> Optional[str]:
        if self.scenario_key is None:
            return None

        params = f"{self.series_fraction}_{self.missing_value_restrictions}_{self.missing_value_adjustment}_{self.use_c}"
        params_hash = hashlib.md5(params.encode()).hexdigest()[:8]

        return f"{self.scenario_key}_{params_hash}"
