import hashlib
import os
from typing import Optional

import dtw_missing.dtw_missing as dtw_m
import h5py
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
        cache_dir: Optional[str] = None,
        scenario_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.series_fraction = series_fraction
        self.missing_value_restrictions = missing_value_restrictions
        self.missing_value_adjustment = missing_value_adjustment
        self.use_c = use_c
        self.cache_dir = cache_dir
        self.scenario_key = scenario_key

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
