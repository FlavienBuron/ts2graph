import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from datasets.dataloaders.graphloader import GraphLoader
from datasets.dataset_registry import DatasetRegistry
from datasets.missingness_scenarios import ScenarioManager


@DatasetRegistry.register("metr_la")
@DatasetRegistry.register("metr-la")
@DatasetRegistry.register("metrla")
@DatasetRegistry.register("la")
class MetrLALoader(GraphLoader):
    def __init__(
        self,
        dataset_path: str = "./datasets/data/metr_la",
        impute_zeros: bool = True,
        impute_nans: bool = True,
        nan_method: str = "mean",
        freq: str = "5min",
        masked_sensors: list | None = None,
        window: int = 12,
        missingness: Optional[Dict] = None,
        **kwargs,
    ):
        self._scenario = None
        self.dataset_path = dataset_path
        self.impute_zeros = impute_zeros

        # Metr-LA covers March-June 2012 (~4 months)
        # Standard split: June for test, May for validation
        self.test_months = [6]  # June
        self.infer_eval_from = "next"

        self.missingness_config = missingness or {}

        data_raw, distances, eval_mask_loaded = self.load_raw()

        if self.missingness_config.get("enabled", False):
            # INJECTION MODE
            data, missing_mask, eval_mask, distances = self._load_with_missingness(
                data_raw=data_raw,
                distances=distances,
                impute_nans=impute_nans,
                impute_zeros=impute_zeros,
                masked_sensors=masked_sensors,
            )
        else:
            # BASELINE MODE
            data, missing_mask, distances = self._default_load(
                data_raw=data_raw,
                distances=distances,
                loaded_eval_mask=eval_mask_loaded,
                impute_nans=impute_nans,
                impute_zeros=impute_zeros,
                masked_sensors=masked_sensors,
            )
            eval_mask = self.eval_mask

        self.distances = distances
        self.masked_sensors = list(masked_sensors) if masked_sensors is not None else list()

        super().__init__(
            dataframe=data,
            missing_mask=missing_mask,
            eval_mask=eval_mask,
            freq=freq,
            aggr="nearest",
            window=window,
            scenario=self._scenario,
        )

    def load_raw(self, small: bool = False) -> Tuple[pd.DataFrame, np.ndarray, Optional[pd.DataFrame]]:
        if small:
            print("[WARN] Small dataset for Metr-LA isn't implemented, continuing with full size")
        path = os.path.join(self.dataset_path, "metr_la.h5")
        df = pd.read_hdf(path)

        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq="5min")
        df = df.reindex(index=date_range)

        distances = self._load_distance_matrix()

        eval_mask = None

        return df, distances, eval_mask

    def _load_distance_matrix(self) -> np.ndarray:
        path = os.path.join(self.dataset_path, "metr_la_dist.npy")
        try:
            dist = np.load(path)
        except FileNotFoundError:
            print("Distance matrix not found. Computing from CSV...")
            distances_csv = pd.read_csv(os.path.join(self.dataset_path, "distances_la.csv"))
            with open(os.path.join(self.dataset_path, "sensor_ids_la.txt")) as f:
                ids = f.read().strip().split(",")

            num_sensors = len(ids)
            dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf
            sensor_id_to_ind = {int(sensor_id): i for i, sensor_id in enumerate(ids)}

            for row in distances_csv.values:
                if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                    continue
                dist[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

            np.save(path, dist)
            print(f"Saved distance matrix to {path}")

        return dist

    def _default_load(
        self,
        data_raw: pd.DataFrame,
        distances: np.ndarray,
        loaded_eval_mask: Optional[pd.DataFrame],
        impute_nans: bool = True,
        impute_zeros: bool = True,
        masked_sensors: Optional[List[int]] = None,
    ):
        mask = ~np.isnan(data_raw.values)
        if impute_zeros:
            mask = mask & (data_raw.values != 0.0)
        missing_mask = mask.astype(bool)

        if loaded_eval_mask is None:
            print("Inferring eval mask")
            eval_mask = self._infer_mask(data_raw)
        else:
            eval_mask = loaded_eval_mask
        eval_mask = eval_mask.values.astype(bool)

        if masked_sensors is not None and len(masked_sensors) > 0:
            eval_mask[:, masked_sensors] = np.where(missing_mask[:, masked_sensors], True, False)
        self.eval_mask = eval_mask

        data = data_raw.copy()
        if impute_zeros:
            data = data.replace(0.0, np.nan)
        if impute_nans:
            data = data.fillna(self._compute_mean(data))

        return data, missing_mask, distances

    def _load_with_missingness(
        self,
        data_raw: pd.DataFrame,
        distances: np.ndarray,
        impute_nans: bool,
        impute_zeros: bool,
        masked_sensors: Optional[List[int]],
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
        """Load with synthetic missingness injection via ScenarioManager."""
        # Build baseline mask
        baseline_mask = ~np.isnan(data_raw.values)
        if impute_zeros:
            baseline_mask = baseline_mask & (data_raw.values != 0.0)
        baseline_missing_rate = 1.0 - baseline_mask.mean()

        pattern = self.missingness_config.get("pattern", "mcar_blocks")
        is_aligned = pattern in ("aligned_blocks", "aligned")
        eval_mask_mode = self.missingness_config.get("eval_mask_mode", "fixed")

        if is_aligned:
            coverage_levels = self.missingness_config.get("target_rate", [0.3])
            target_rate = coverage_levels[0] if isinstance(coverage_levels, list) else float(coverage_levels)
            first_rate = self.missingness_config.get("first_rate", 0.1)
            is_first_rate = target_rate == first_rate
            eval_fraction = None
        else:
            target_rates = self.missingness_config.get("target_rate", 0.40)
            target_rate = target_rates[0] if isinstance(target_rates, list) else float(target_rates)
            first_rate = self.missingness_config.get("first_rate", 0.3)
            is_first_rate = target_rate == first_rate
            eval_fraction = None if is_first_rate else max(0.0, first_rate - baseline_missing_rate)

        aligned_kwargs = {}
        if is_aligned:
            aligned_kwargs["data_index"] = pd.DatetimeIndex(data_raw.index)
            aligned_kwargs["sensor_fraction"] = float(target_rate)
            aligned_kwargs["sensor_pattern"] = self.missingness_config.get("sensor_pattern", "random")
            aligned_kwargs["placement"] = self.missingness_config.get("placement", "span_all")
            aligned_kwargs["test_months"] = self.missingness_config.get("test_months", [6])

        # Initialize injection manager
        scenario_manager = ScenarioManager(cache_dir=self.missingness_config.get("cache_dir", "./datasets/missingness_scenarios/cache"))
        scenario_manager.set_data_hash(data_raw.values)
        scenario_manager.set_original_missing_from_mask(mask=baseline_mask)

        rate_type = "sensor coverage" if is_aligned else "missing rate"
        print(f"🔧 Retrieving missingness scenario: {target_rate:.0%} ({rate_type})")
        print(f"   Pattern: {pattern} | Block size: {self.missingness_config.get('block_size', 12)}")
        print(f"   Eval mask mode: {eval_mask_mode}")
        print(f"[DEBUG] {self.missingness_config=}")

        # Generate scenario
        scenario = scenario_manager.get_scenario(
            shape=data_raw.shape,
            base_missing_rate=baseline_missing_rate,
            target_rate=target_rate,
            pattern=self.missingness_config.get("pattern", "mcar_blocks"),
            block_size=self.missingness_config.get("block_size", 12),
            seed=self.missingness_config.get("seed", 42),
            cumulative=self.missingness_config.get("cumulative", False),
            force_regenerate=self.missingness_config.get("force_regenerate", False),
            eval_fraction=eval_fraction,
            is_first_rate=is_first_rate,
            **aligned_kwargs,
        )

        self._scenario = scenario

        # Select eval mask based on mode
        if eval_mask_mode == "fixed":
            eval_mask = scenario.eval_mask_fixed
        elif eval_mask_mode == "newly":
            eval_mask = scenario.eval_mask_newly
        elif eval_mask_mode == "cumulative":
            eval_mask = scenario.eval_mask_cumulative
        else:
            raise ValueError(f"Unknown eval_mask_mode: {eval_mask_mode}")

        # Log scenario info
        meta = scenario.metadata
        print(f"   Actual missing: {meta.get('actual_rate', 'N/A'):.4%}")
        print(f"   Eval targets: {eval_mask.sum():,} ({eval_mask.mean():.4%} of data)")
        print(f"   Eval fraction: {meta.get('eval_fraction', 'N/A'):.4%}")
        print(f"   Injection mode: {meta.get('injection_mode', 'N/A')}")

        # Handle masked_sensors override
        if masked_sensors is not None and len(masked_sensors) > 0:
            eval_mask[:, masked_sensors] = np.where(baseline_mask[:, masked_sensors].astype(bool), True, False)

        # Impute original NaNs for model input
        data_imputed = data_raw.copy()
        if impute_zeros:
            data_imputed = data_imputed.replace(0.0, np.nan)
        if impute_nans:
            data_imputed = data_imputed.fillna(self._compute_mean(data_imputed))

        self.eval_mask = eval_mask.astype(int)

        return (
            data_imputed,
            scenario.full_mask.astype(bool),
            eval_mask.astype(bool),
            distances,
        )

    def grin_split(
        self,
        val_len: float = 0.1,
        test_len: float = 0.2,
        in_sample: bool = True,
        window: int = 12,
    ):
        """
        Standard chronological percentage-based split for traffic datasets (Metr-LA/PEMS-BAY).
        Replaces the month-based split used in AirQ.
        """
        idx = np.arange(len(self))

        # Calculate absolute lengths based on percentages
        if test_len < 1:
            test_len_abs = int(test_len * len(idx))
        else:
            test_len_abs = int(test_len)

        if val_len < 1:
            val_len_abs = int(val_len * (len(idx) - test_len_abs))
        else:
            val_len_abs = int(val_len)

        test_start = len(idx) - test_len_abs
        val_start = test_start - val_len_abs

        # No `- window` subtraction needed here because len(self) already
        # returns the number of valid sliding window samples, not raw timestamps.

        train_idxs = idx[:val_start]
        val_idxs = idx[val_start:test_start]
        test_idxs = idx[test_start:]

        return train_idxs, val_idxs, test_idxs

    def _infer_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        """Infer evaluation mask from data pattern (same logic as AirQ)."""
        observed_mask = data.isna().astype("bool")
        if self.impute_zeros:
            observed_mask = observed_mask | (data == 0.0)

        eval_mask = pd.DataFrame(index=data.index, columns=data.columns, data=0).astype("bool")

        if self.infer_eval_from == "previous":
            offset = -1
        elif self.infer_eval_from == "next":
            offset = 1
        else:
            raise ValueError("infer_eval_mask can only be one of ['previous', 'next']")

        months = sorted(set(zip(data.index.year, data.index.month)))
        length = len(months)

        for i in range(length):
            j = (i + offset) % length
            year_i, month_i = months[i]
            year_j, month_j = months[j]

            mask_j = observed_mask[(data.index.year == year_j) & (data.index.month == month_j)]
            mask_i = mask_j.shift(1, pd.DateOffset(months=12 * (year_i - year_j) + (month_i - month_j)))
            mask_i = mask_i[~mask_i.index.duplicated(keep="first")]
            mask_i = mask_i[np.isin(mask_i.index, data.index)]

            eval_mask.loc[mask_i.index] = (~mask_i.loc[mask_i.index].astype(bool) & data.loc[mask_i.index].astype(bool)).astype(eval_mask.dtypes.iloc[0])

        return eval_mask

    def _compute_mean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean imputation values adapted for 5-minute frequency data.
        Groups by day-of-week, hour, and minute for better temporal patterns.
        """
        data_mean = data.copy()

        # For 5-minute data: group by (dayofweek, hour, minute), then (hour, minute), etc.
        condition0 = [
            data_mean.index.dayofweek,
            data_mean.index.hour,
            data_mean.index.minute,
        ]
        condition1 = [
            data_mean.index.hour,
            data_mean.index.minute,
        ]
        conditions = [condition0, condition1, condition1[1:]]

        while data_mean.isna().values.sum() and len(conditions):
            nan_mean = data_mean.groupby(conditions[0]).transform("mean")
            data_mean = data_mean.fillna(nan_mean)
            conditions = conditions[1:]

        if data_mean.isna().values.sum():
            data_mean = data_mean.ffill()
            data_mean = data_mean.bfill()

        return data_mean

    def _disjoint_months(
        self,
        months: List = [],
        sync_mode: str = "window",
    ):
        """Split indices into those within specified months vs. others."""
        idxs = np.arange(len(self))

        if sync_mode == "window":
            start, end = 0, self.window - 1
        elif sync_mode == "horizon":
            horizon_offset = self.horizon_offset
            start, end = horizon_offset, horizon_offset + self.horizon - 1
        else:
            raise ValueError(f"Invalid sync mode type: {sync_mode}. Expected 'window' or 'horizon'")

        if self.index is not None:
            # After idxs (in test months)
            start_in_months = np.isin(self.index[self._indices + start].month, months)
            end_in_months = np.isin(self.index[self._indices + end].month, months)
            idxs_in_months = start_in_months & end_in_months
            after_idxs = idxs[idxs_in_months]

            # Before idxs (not in test months)
            months_before = np.setdiff1d(np.arange(1, 13), months)
            start_in_months = np.isin(self.index[self._indices + start].month, months_before)
            end_in_months = np.isin(self.index[self._indices + end].month, months_before)
            idxs_in_months = start_in_months & end_in_months
            prev_idxs = idxs[idxs_in_months]

            return prev_idxs, after_idxs
        else:
            raise ValueError("Index not initialized")

    def overlapping_indices(self, idxs1, idxs2, sync_mode="window", as_mask=False) -> tuple[np.ndarray, np.ndarray]:
        """Find overlapping timestamps between two index sets."""
        assert sync_mode in ["window", "horizon"], "sync_mode can only be 'window' or 'horizon'"
        timestamp1 = self.data_timestamps(idxs1, flatten=False)[sync_mode]
        timestamp2 = self.data_timestamps(idxs2, flatten=False)[sync_mode]
        common_timestamps = np.intersect1d(np.unique(timestamp1), np.unique(timestamp2))
        is_overlapping = lambda sample: np.any(np.isin(sample, common_timestamps))
        m1 = np.apply_along_axis(is_overlapping, 1, timestamp1)
        m2 = np.apply_along_axis(is_overlapping, 1, timestamp2)
        if as_mask:
            return m1, m2
        return np.sort(idxs1[m1]), np.sort(idxs2[m2])

    def expand_indices(self, indices=None, unique=False) -> Dict:
        """Expand sample indices to cover full window/horizon ranges."""
        ds_indices = dict.fromkeys([time for time in ["window", "horizon"] if getattr(self, time) > 0])
        indices = np.arange(len(self._indices)) if indices is None else indices

        if "window" in ds_indices:
            window_idxs = [np.arange(idx, idx + self.window) for idx in self._indices[indices]]
            ds_indices["window"] = np.concatenate(window_idxs)

        if "horizon" in ds_indices:
            horizon_idxs = [np.arange(idx + self.horizon_offset, idx + self.horizon_offset + self.horizon) for idx in self._indices[indices]]
            ds_indices["horizon"] = np.concatenate(horizon_idxs)

        if unique:
            ds_indices = {k: np.unique(v) for k, v in ds_indices.items() if v is not None}

        return ds_indices

    def data_timestamps(self, indices=None, flatten=True) -> Dict:
        """Get actual timestamps for given indices."""
        ds_indices = self.expand_indices(indices, unique=False)
        ds_timestamp = {k: self.index[v] for k, v in ds_indices.items()}
        if not flatten:
            ds_timestamp = {k: np.array(v).reshape(-1, getattr(self, k)) for k, v in ds_timestamp.items()}
        return ds_timestamp
