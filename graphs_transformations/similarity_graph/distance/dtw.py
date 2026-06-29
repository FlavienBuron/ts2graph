import hashlib
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

import dtw_missing.dtw_missing as dtw_m
import h5py
import numpy as np
import torch

from ..specs.registry import register_distance
from .base import DistanceFunction

# # ============================================================
# # Worker global state
# # ============================================================
# _X_WORKER = None
# _MASK_WORKER = None


def init_worker(X_np: np.ndarray, mask_np: Optional[np.ndarray]):
    global _X_WORKER, _MASK_WORKER
    _X_WORKER = X_np
    _MASK_WORKER = mask_np


def _compute_dtw_row(args: Tuple) -> Tuple[int, List[Tuple[int, float]]]:
    i, N, F, window_len, restrictions, adjustment, use_c = args

    xi_np = _X_WORKER[:, i, :]
    mi_np = _MASK_WORKER[:, i, :] if _MASK_WORKER is not None else None

    xi = xi_np.copy() if mi_np is not None else xi_np
    if mi_np is not None:
        xi[~mi_np] = np.nan

    if F == 1:
        xi = xi.squeeze(-1)

    row_results = []

    for j in range(i + 1, N):
        xj_np = _X_WORKER[:, j, :]
        mj_np = _MASK_WORKER[:, j, :] if _MASK_WORKER is not None else None

        xj = xj_np.copy() if mj_np is not None else xj_np
        if mj_np is not None:
            xj[~mj_np] = np.nan

        if F == 1:
            xj = xj.squeeze(-1)

        cost = dtw_m.warping_paths(
            s1=xi,
            s2=xj,
            window=window_len,
            missing_value_restrictions=restrictions,
            missing_value_adjustment=adjustment,
            use_c=use_c,
        )[0]

        row_results.append((j, cost))

    return i, row_results


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

        # T, N, F = X.shape
        # window_len = max(1, int(self.series_fraction * T))
        #
        # # Convert to NumPy ONCE to avoid pickling overhead in multiprocessing
        # X_np = np.ascontiguousarray(X.cpu().numpy())
        # mask_np = np.ascontiguousarray(mask.cpu().numpy()) if mask is not None else None
        #
        # # Pre-extract 1D slices for each node.
        # # This prevents passing the massive 3D array to every worker process.
        # X_list = [X_np[:, i, :] for i in range(N)]
        # mask_list = [mask_np[:, i, :] for i in range(N)] if mask_np is not None else [None] * N
        #
        # # Build the task list
        # tasks = []
        # for i in range(N):
        #     for j in range(i + 1, N):
        #         tasks.append(
        #             (
        #                 i,
        #                 j,
        #                 X_list[i],
        #                 X_list[j],
        #                 mask_list[i],
        #                 mask_list[j],
        #                 window_len,
        #                 self.missing_value_restrictions,
        #                 self.missing_value_adjustment,
        #                 self.use_c,
        #                 F,
        #             )
        #         )
        # print(f"Computing {len(tasks):,} pairs using {self.n_jobs} CPU cores...")
        #
        # D = torch.full((N, N), float("inf"))
        #
        # # Use ProcessPoolExecutor to bypass the Python GIL
        # with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
        #     # executor.map preserves order, but we just need to unpack the results
        #     for i, j, cost in executor.map(_compute_dtw_pair, tasks):
        #         D[i, j] = cost
        #         D[j, i] = cost

        T, N, F = X.shape
        window_len = max(1, int(self.series_fraction * T))

        # Convert to contiguous NumPy arrays ONCE
        X_np = np.ascontiguousarray(X.cpu().numpy())
        mask_np = np.ascontiguousarray(mask.cpu().numpy()) if mask is not None else None

        # Create exactly N tasks (437 tasks).
        # Notice we DO NOT pass X_np or mask_np here. They are passed via initializer.
        tasks = [(i, N, F, window_len, self.missing_value_restrictions, self.missing_value_adjustment, self.use_c) for i in range(N)]

        print(f"Computing {N * (N - 1) // 2:,} pairs across {N} row-tasks using {self.n_jobs} CPU cores...")

        D = torch.full((N, N), float("inf"))

        # ====================================================
        # Progress tracking
        # ====================================================
        start_time = time.perf_counter()
        last_print = start_time
        completed_rows = 0
        total_rows = N
        total_pairs = N * (N - 1) // 2
        completed_pairs = 0

        # The initializer passes the heavy arrays to each worker process exactly ONCE.
        with ProcessPoolExecutor(max_workers=self.n_jobs, initializer=init_worker, initargs=(X_np, mask_np)) as executor:
            for i, row_results in executor.map(_compute_dtw_row, tasks):
                for j, cost in row_results:
                    D[i, j] = cost
                    D[j, i] = cost
                    completed_pairs += 1

                completed_rows += 1

                now = time.perf_counter()
                if now - last_print >= 20:  # every 20s
                    elapsed = now - start_time
                    pair_rate = completed_pairs / elapsed if elapsed > 0 else 0
                    eta = (total_pairs - completed_pairs) / pair_rate if pair_rate > 0 else float("inf")

                    print(
                        f"[DTW] rows {completed_rows}/{total_rows} "
                        f"({completed_rows / total_rows:.1%}) | "
                        f"pairs {completed_pairs:,}/{total_pairs:,} "
                        f"({completed_pairs / total_pairs:.1%}) | "
                        f"{pair_rate:.1f} pairs/s | "
                        f"ETA {eta / 60:.1f} min"
                    )

                    last_print = now
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
