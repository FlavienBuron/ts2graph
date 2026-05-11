import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

from datasets.dataloaders.graphloader import GraphLoader
from datasets.dataset_registry import DatasetRegistry
from datasets.missingness_scenarios import ScenarioManager

EARTH_RADIUS = 6371.0088


@DatasetRegistry.register("airq", "small")
@DatasetRegistry.register("airq", "default")
class AirQualityLoader(GraphLoader):
    def __init__(
        self,
        dataset_path: str = "./datasets/data/air_quality/",
        small: bool = False,
        impute_nans: bool = True,
        nan_method: str = "mean",
        freq: str = "60min",
        masked_sensors: list | None = None,
        window: int = 36,
        missingness: Optional[Dict] = None,
        **kwargs,
    ):
        self._scenario = None
        self.dataset_path = dataset_path

        self.test_months = [3, 6, 9, 12]
        self.infer_eval_from = "next"

        self.missingness_config = missingness or {}

        data_raw, stations, eval_mask_loaded = self.load_raw(small)

        if self.missingness_config.get("enabled", False):
            # INJECTION MODE: Apply/retrieve missingness scenario
            data, missing_mask, eval_mask, distances = self._load_with_missingness(
                data_raw=data_raw,
                stations=stations,
                impute_nans=impute_nans,
                masked_sensors=masked_sensors,
            )
        else:
            # BASELINE MODE: Load as before (backward compatible)
            data, missing_mask, distances = self._default_load(
                data_raw=data_raw,
                stations=stations,
                loaded_eval_mask=eval_mask_loaded,
                impute_nans=impute_nans,
                masked_sensors=masked_sensors,
            )
            eval_mask = self.eval_mask  # Set by _load_baseline

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

    def load_raw(self, small: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        if small:
            path = os.path.join(self.dataset_path, "small36.h5")
            eval_mask = pd.DataFrame(pd.read_hdf(path, "eval_mask"))
        else:
            path = os.path.join(self.dataset_path, "full437.h5")
            eval_mask = None
        data = pd.DataFrame(pd.read_hdf(path, key="pm25"))
        stations = pd.DataFrame(pd.read_hdf(path, key="stations"))
        return data, stations, eval_mask

    def _default_load(
        self,
        data_raw: pd.DataFrame,
        stations: pd.DataFrame,
        loaded_eval_mask: Optional[pd.DataFrame],
        impute_nans: bool = True,
        small: bool = False,
        masked_sensors: list | None = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        data, stations, eval_mask = self.load_raw(small=small)
        missing_mask = (~np.isnan(data_raw.values)).astype("bool")  # 0=missing, 1=observed
        if loaded_eval_mask is None:
            print("Infering eval mask")
            eval_mask = self._infer_mask(data)
        else:
            eval_mask = loaded_eval_mask
        eval_mask = eval_mask.values.astype("bool")
        if masked_sensors is not None and len(masked_sensors) > 0:
            eval_mask[:, masked_sensors] = np.where(missing_mask[:, masked_sensors], True, False)
        self.eval_mask = eval_mask
        if impute_nans:
            data = data.fillna(self._compute_mean(data))
        else:
            data = data_raw.copy()

        stations_coords = stations.loc[:, ["latitude", "longitude"]]
        dist = self._geographical_distance(stations_coords)
        return data, missing_mask, dist

    def _load_with_missingness(
        self,
        data_raw: pd.DataFrame,
        stations: pd.DataFrame,
        impute_nans: bool,
        masked_sensors: Optional[List[int]],
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
        baseline_mask = ~np.isnan(data_raw.values)
        baseline_missing_rate = 1.0 - baseline_mask.mean()

        pattern = self.missingness_config.get("pattern", "mcar_blocks")
        is_aligned = pattern in ("aligned_blocks", "aligned")
        eval_mask_mode = self.missingness_config.get("eval_mask_mode", "fixed")

        if is_aligned:
            # Aligned blocks: target = sensor coverage fraction
            coverage_levels = self.missingness_config.get("target_rate", [0.3])
            target_rate = coverage_levels[0] if isinstance(coverage_levels, list) else float(coverage_levels)

            # Eval fraction is a direct upper-bound from config (not derived from rates)
            eval_fraction = self.missingness_config.get("eval_fraction", 0.047)
            is_first_rate = True  # Eval logic is independent of baseline for aligned
        else:
            # MCAR: target = global missing rate
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
            aligned_kwargs["test_months"] = self.missingness_config.get("test_months", [3, 6, 9, 12])

        # Initialize injection manager
        scenario_manager = ScenarioManager(cache_dir=self.missingness_config.get("cache_dir", "./datasets/missingness_scenarios/cache"))
        scenario_manager.set_data_hash(data_raw.values)
        scenario_manager.set_original_missing_from_mask(mask=baseline_mask)

        rate_type = "sensor coverage" if is_aligned else "missing rate"
        print(f"🔧 Retrieving missingness scenario: {target_rate:.0%} ({rate_type})")
        print(f"   Pattern: {pattern} | Block size: {self.missingness_config.get('block_size', 10)}")
        print(f"   Eval mask mode: {eval_mask_mode}")
        print(f"[DEBUG] {self.missingness_config=}")
        # Generate scenario
        scenario = scenario_manager.get_scenario(
            shape=data_raw.shape,
            base_missing_rate=baseline_missing_rate,
            target_rate=target_rate if not is_aligned else 0.0,
            pattern=self.missingness_config.get("pattern", "mcar_blocks"),
            block_size=self.missingness_config.get("block_size", 10),
            seed=self.missingness_config.get("seed", 42),
            cumulative=self.missingness_config.get("cumulative", False),
            force_regenerate=self.missingness_config.get("force_regenerate", False),
            eval_fraction=eval_fraction,
            is_first_rate=is_first_rate,
            **aligned_kwargs,
        )

        # Store scenario reference for main.py to access all eval masks
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
        print(f"   Actual missing: {meta.get('actual_rate', 'N/A'):.2%}")
        print(f"   Eval targets: {eval_mask.sum():,} ({eval_mask.mean():.2%} of data)")
        print(f"   Eval fraction: {meta.get('eval_fraction', 'N/A'):.1%}")
        print(f"   Injection mode: {meta.get('injection_mode', 'N/A')}")

        # Handle masked_sensors override
        if masked_sensors is not None and len(masked_sensors) > 0:
            eval_mask[:, masked_sensors] = np.where(baseline_mask[:, masked_sensors].astype(bool), True, False)

        # Impute original NaNs for model input
        if impute_nans:
            data_imputed = data_raw.fillna(self._compute_mean(data_raw))
        else:
            data_imputed = data_raw.copy()

        # Store eval_mask for later access
        self.eval_mask = eval_mask.astype(int)

        # Compute distances
        stations_coords = stations.loc[:, ["latitude", "longitude"]]
        distances = self._geographical_distance(stations_coords)

        return (
            data_imputed,
            scenario.full_mask.astype(bool),
            eval_mask.astype(bool),
            distances,
        )

    def grin_split(
        self,
        val_len: float = 0.1,
        in_sample: bool = False,
        window: int = 36,
    ):
        nontest_idxs, test_idxs = self._disjoint_months(months=self.test_months, sync_mode="horizon")
        if in_sample:
            train_idxs = np.arange(len(self))
            val_months = [(m - 1) % 12 for m in self.test_months]
            _, val_idxs = self._disjoint_months(months=val_months, sync_mode="horizon")
        else:
            val_len = (int(val_len * len(nontest_idxs)) if val_len < 1 else val_len) // len(self.test_months)
            # get indices of first day of each testing month
            delta_idxs = np.diff(test_idxs)
            end_month_idxs = test_idxs[1:][np.flatnonzero(delta_idxs > delta_idxs.min())]
            if len(end_month_idxs) < len(self.test_months):
                end_month_idxs = np.insert(end_month_idxs, 0, test_idxs[0])
            # expand month indices
            month_val_idxs = [np.arange(v_idx - val_len, v_idx) - window for v_idx in end_month_idxs]
            val_idxs = np.concatenate(month_val_idxs) % len(self)
            # remove overlapping indices from training set
            ovl_idxs, _ = self.overlapping_indices(nontest_idxs, val_idxs, sync_mode="horizon", as_mask=True)
            train_idxs = nontest_idxs[~ovl_idxs]
        return train_idxs, val_idxs, test_idxs

    def _geographical_distance(self, coords: pd.DataFrame, to_rad: bool = True) -> pd.DataFrame:
        """
        Compute the geographical distance between coordinates points
        """

        _AVG_EARTH_RADIUS_KM = 6371.0088

        coords_pairs = coords.values
        if to_rad:
            coords_pairs = np.vectorize(np.radians)(coords_pairs)
        dist = haversine_distances(coords_pairs) * _AVG_EARTH_RADIUS_KM
        dist_df = pd.DataFrame(dist, coords.index, coords.index)
        return dist_df

    def _infer_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        observed_mask = data.isna().astype("bool")
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
        data_mean = data.copy()

        condition0 = [
            data_mean.index.year,
            data_mean.index.isocalendar().week,
            data_mean.index.hour,
        ]
        condition1 = [
            data_mean.index.year,
            data_mean.index.month,
            data_mean.index.hour,
        ]
        conditions = [condition0, condition1, condition1[1:], condition1[2:]]
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
        idxs = np.arange(len(self))
        if sync_mode == "window":
            start, end = 0, self.window - 1
        elif sync_mode == "horizon":
            horizon_offset = self.horizon_offset
            start, end = horizon_offset, horizon_offset + self.horizon - 1
        else:
            raise ValueError(f"Invalid sync mode type: {sync_mode}. Expected 'window' or 'horizon'")
        if self.index is not None:
            # after idxs
            start_in_months = np.isin(self.index[self._indices + start].month, months)
            end_in_months = np.isin(self.index[self._indices + end].month, months)
            idxs_in_months = start_in_months & end_in_months
            after_idxs = idxs[idxs_in_months]

            # before idxs
            months_before = np.setdiff1d(np.arange(1, 13), months)
            start_in_months = np.isin(self.index[self._indices + start].month, months_before)
            end_in_months = np.isin(self.index[self._indices + end].month, months_before)
            idxs_in_months = start_in_months & end_in_months
            prev_idxs = idxs[idxs_in_months]

            return prev_idxs, after_idxs
        else:
            raise ValueError

    def overlapping_indices(self, idxs1, idxs2, sync_mode="window", as_mask=False) -> tuple[np.ndarray, np.ndarray]:
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
        ds_indices = dict.fromkeys([time for time in ["window", "horizon"] if getattr(self, time) > 0])
        indices = np.arange(len(self._indices)) if indices is None else indices
        if "window" in ds_indices:
            window_idxs = [np.arange(idx, idx + self.window) for idx in self._indices[indices]]
            ds_indices["window"] = np.concatenate(window_idxs)
        if "horizon" in ds_indices:
            horizon_idxs = [
                np.arange(
                    idx + self.horizon_offset,
                    idx + self.horizon_offset + self.horizon,
                )
                for idx in self._indices[indices]
            ]
            ds_indices["horizon"] = np.concatenate(horizon_idxs)
        if unique:
            ds_indices = {k: np.unique(v) for k, v in ds_indices.items() if v is not None}
        return ds_indices

    def data_timestamps(self, indices=None, flatten=True) -> Dict:
        ds_indices = self.expand_indices(indices, unique=False)
        ds_timestamp = {k: self.index[v] for k, v in ds_indices.items()}
        if not flatten:
            ds_timestamp = {k: np.array(v).reshape(-1, getattr(self, k)) for k, v in ds_timestamp.items()}
        return ds_timestamp
