import hashlib
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import dtw_missing.dtw_missing as dtw_m
import h5py
import numpy as np
import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


def init_worker(X_np, mask_np):
    global _X_WORKER, _MASK_WORKER
    _X_WORKER = X_np
    _MASK_WORKER = mask_np


def _compute_dtw_pair(args):
    i, j, window_len, restrictions, adjustment, use_c, F = args

    xi = _X_WORKER[i]
    xj = _X_WORKER[j]

    if _MASK_WORKER is not None:
        mi = _MASK_WORKER[i]
        mj = _MASK_WORKER[j]

        xi = xi.copy()
        xj = xj.copy()

        xi[~mi] = np.nan
        xj[~mj] = np.nan

    if F == 1:
        xi = xi.reshape(-1)
        xj = xj.reshape(-1)

    cost = dtw_m.warping_paths(
        s1=xi,
        s2=xj,
        window=window_len,
        missing_value_restrictions=restrictions,
        missing_value_adjustment=adjustment,
        use_c=use_c,
    )[0]

    return i, j, float(cost)


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

        X_np = np.ascontiguousarray(X.cpu().numpy())
        mask_np = np.ascontiguousarray(mask.cpu().numpy()) if mask is not None else None

        tasks = [(i, j, window_len, self.missing_value_restrictions, self.missing_value_adjustment, self.use_c, F) for i in range(N) for j in range(i + 1, N)]

        print(f"Computing {len(tasks):,} pairs using {self.n_jobs} cores...")

        D = torch.full((N, N), float("inf"))

        start = time.perf_counter()
        last_report = start
        completed = 0
        total = len(tasks)

        with ProcessPoolExecutor(
            max_workers=self.n_jobs,
            initializer=init_worker,
            initargs=(X_np, mask_np),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = {executor.submit(_compute_dtw_pair, t): t for t in tasks}

            for future in as_completed(futures):
                i, j, cost = future.result()
                D[i, j] = cost
                D[j, i] = cost

                completed += 1

                now = time.perf_counter()
                if now - last_report > 30:
                    elapsed = now - start
                    rate = completed / elapsed
                    eta = (total - completed) / rate if rate > 0 else float("inf")

                    print(f"{completed:,}/{total:,} ({completed / total * 100:.1f}%) | {rate:.2f} pairs/s | ETA {eta / 60:.1f} min")

                    last_report = now
                    # for i, j, cost in executor.map(_compute_dtw_pair, tasks, chunksize=128):
                    #     D[i, j] = cost
                    #     D[j, i] = cost

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
