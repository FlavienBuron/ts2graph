from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.utils.data import Dataset

from utils.helpers import debug_mask_relationship


class GraphLoader(Dataset, ABC):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        missing_mask: np.ndarray,
        eval_mask: np.ndarray | torch.Tensor | None = None,
        freq: str | None = None,
        aggr: str = "sum",
        exogenous=None,
    ) -> None:
        self.eval_mask = eval_mask
        self._exogenous_keys = dict()
        self._reserved_signature = {"data", "trend", "x", "y"}

        # reproduce 'pd_dataset' class from GRIN
        self._store_pandas_data(
            dataframe=dataframe,
            mask=missing_mask,
            freq=freq,
            aggr=aggr,
        )
        debug_mask_relationship(
            torch.tensor(self.mask), torch.tensor(self.eval_mask), "GraphLoader mask"
        )
        debug_mask_relationship(
            torch.tensor(self.training_mask),
            torch.tensor(self.eval_mask),
            "GraphLoader mask",
        )

        # Emulate GRIN's SpatioaTemporal classes, into one
        self.data, self.index = self.as_numpy(return_idx=True)
        if self.index is None:
            raise AttributeError("Dataset index is returned as None")

        # self.mask = self.training_mask

        if exogenous is None:
            exogenous = dict()
        exogenous["mask_window"] = (
            self.training_mask.detach().clone()
            if isinstance(self.training_mask, torch.Tensor)
            else torch.tensor(self.training_mask)
        )
        if eval_mask is not None:
            exogenous["eval_mask_window"] = torch.tensor(eval_mask)
        for name, value in exogenous.items():
            self._add_exogenous(value, name, for_window=True, for_horizon=True)

        try:
            freq = freq or self.index.freq or self.index.inferred_freq
            self.freq = pd.tseries.frequencies.to_offset(freq)
        except AttributeError:
            self.freq = None

        self.trend = None
        self.scaler = None

        self.horizon = 36
        self.window = 36
        self.delay = -self.window
        self.stride = 1

        self._indices = np.arange(self.df.shape[0] - self.sample_span + 1)[
            :: self.stride
        ]

        self.scaler = None

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, item: int) -> Any:
        return self.get(item, self.preprocess)

    def __contains__(self, item):
        return item in self._exogenous_keys

    @property
    def training_mask(self):
        return self.mask if self.eval_mask is None else (self.mask & ~self.eval_mask)

    @property
    def has_mask(self):
        return self._mask is not None

    @property
    def shape(self):
        return self.df.values.shape

    @property
    def mask(self):
        if self.has_mask:
            return self._mask
        return np.zeros_like(self.shape).astype("bool")

    @mask.setter
    def mask(self, value: np.ndarray | torch.Tensor):
        assert value is not None
        # Convert to tensor if needed, using proper method
        if isinstance(value, torch.Tensor):
            # Already a tensor - use detach().clone()
            tensor_value = value.detach().clone()
        else:
            # Numpy array - convert to tensor
            tensor_value = torch.from_numpy(value)
        self._mask = self._check_input(tensor_value)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        assert value is not None
        self._data = self._check_input(torch.tensor(value))

    @property
    def trend(self):
        return self._trend

    @trend.setter
    def trend(self, value):
        self._trend = self._check_input(value)

    @property
    def horizon_offset(self):
        return self.window + self.delay

    @property
    def sample_span(self):
        return max(self.horizon_offset + self.horizon, self.window)

    @property
    def preprocess(self):
        return (self.trend is not None) or (self.scaler is not None)

    @property
    def n_steps(self):
        return self.data.shape[0]

    @property
    def n_channels(self):
        return self.data.shape[-1]

    @property
    def n_nodes(self):
        return self.data.shape[1]

    @property
    def indices(self):
        return self._indices

    # @property
    # def signature(self):
    #     attrs = []
    #     if self.window > 0:
    #         attrs.append("x")
    #         for attr in self._exo_window_keys:
    #             attrs.append(
    #                 attr if attr not in self._exo_common_keys else (attr + "_window")
    #             )
    #     for attr in self._exo_horizon_keys:
    #         attrs.append(
    #             attr if attr not in self._exo_common_keys else (attr + "_horizon")
    #         )
    #     attrs.append("y")
    #     attrs = tuple(attrs)
    #     preprocess = []
    #     if self.trend is not None:
    #         preprocess.append("trend")
    #     if self.scaler is not None:
    #         preprocess.extend(self.scaler.params())
    #     preprocess = tuple(preprocess)
    #     return dict(data=attrs, preprocessing=preprocess)

    @property
    def _exo_window_keys(self):
        return {k for k, v in self._exogenous_keys.items() if v["for_window"]}

    @property
    def _exo_horizon_keys(self):
        return {k for k, v in self._exogenous_keys.items() if v["for_horizon"]}

    @property
    def _exo_common_keys(self):
        return self._exo_window_keys.intersection(self._exo_horizon_keys)

    def get(self, item: int, preprocess: bool = False):
        idx = self._indices[item]
        # print(f"{idx=} {idx+self.window=}")
        res, transform = dict(), dict()
        if self.window > 0:
            res["x"] = self.data[idx : idx + self.window]
            for attr in self._exo_window_keys:
                key = attr if attr not in self._exo_common_keys else (attr + "_window")
                res[key] = getattr(self, attr)[idx : idx + self.window]
            print(
                "eval ⊆ mask            :",
                (~self.mask[1527:1563] & self.eval_mask[1527:1563]).sum().item() == 0,
            )

            # debug_mask_relationship(res["mask"], res["eval_mask"], "get")

        for attr in self._exo_horizon_keys:
            key = attr if attr not in self._exo_common_keys else (attr + "_horizon")
            res[key] = getattr(self, attr)[
                idx + self.horizon_offset : idx + self.horizon_offset + self.horizon
            ]
        res["y"] = self.data[
            idx + self.horizon_offset : idx + self.horizon_offset + self.horizon
        ]
        if preprocess:
            if self.trend is not None:
                y_trend = self.trend[
                    idx + self.horizon_offset : idx + self.horizon_offset + self.horizon
                ]
                res["y"] = res["y"] - y_trend
                transform["trend"] = y_trend
                if "x" in res:
                    res["x"] = res["x"] - self.trend[idx : idx + self.window]
            if self.scaler is not None:
                transform.update(self.scaler.params())
                if "x" in res:
                    res["x"] = self.scaler.transform(res["x"])

        res["x"] = torch.where(res["mask"], res["x"], torch.zeros_like(res["x"]))
        res["mask"] = res["mask"].bool()

        return res, transform

    def _store_pandas_data(
        self,
        dataframe: pd.DataFrame,
        mask: np.ndarray,
        freq: str | None = "60min",
        aggr: str = "sum",
    ):
        self.df = dataframe
        self.index = pd.to_datetime(dataframe.index)

        self.start, self.end = self.index.min(), self.index.max()

        self._mask = mask

        if freq is not None:
            self._resample(freq=freq, aggr=aggr)
        else:
            self.freq = self.index.inferred_freq
            if self.freq is not None:
                self._resample(self.freq, aggr=aggr)
            else:
                raise ValueError("Inferred frequencies returned None")

        self.samples_per_day = int(86400 / pd.Timedelta(self.freq).total_seconds())

    def _resample(self, freq: str, aggr: str):
        resampler = self.df.resample(freq)
        if aggr == "sum":
            self.df = resampler.sum()
        elif aggr == "mean":
            self.df = resampler.mean()
        elif aggr == "nearest":
            self.df = resampler.nearest()
        else:
            raise ValueError(f"{aggr} is not a valid aggregation method")

        if self.has_mask:
            resampler = pd.DataFrame(self._mask, index=self.df.index).resample(freq)
            self._mask = resampler.min().to_numpy()
        self.freq = freq

    def _check_input(self, data: torch.Tensor):
        if data is None:
            # raise ValueError("Data input for dataset should not be None")
            return data
        data = self.check_dim(data)
        data = data.clone().detach()
        if torch.is_floating_point(data):
            return data.float()
        elif data.dtype in [
            torch.int,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            return data.int()
        return data

    def dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def as_numpy(self, return_idx: bool = False):
        if return_idx:
            return self.df.values, self.df.index
        return self.df.values, None

    def _add_exogenous(
        self, exo_data, name: str, for_window: bool = True, for_horizon: bool = False
    ):
        suffix_idx = -7 if "window" in name else -8
        name = name[:suffix_idx]
        for_window = suffix_idx == -7
        for_horizon = suffix_idx == -8
        exo_data = self._check_input(exo_data)
        setattr(self, name, exo_data)
        self._exogenous_keys[name] = dict(
            for_window=for_window, for_horizon=for_horizon
        )

    @abstractmethod
    def get_knn_graph(self, k: float, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_radius_graph(self, radius: float, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_geolocation_graph(self, *args, **kwargs) -> Any:
        pass

    #
    # @abstractmethod
    # def get_dataloader(
    #     self,
    #     test_percent: float = 0.2,
    #     total_missing_percent: float = 0.4,
    #     mask_pattern: str = "default",
    #     shuffle: bool = False,
    #     batch_size: int = 128,
    # ) -> Any:
    #     pass

    @abstractmethod
    def load_raw(self) -> Any:
        pass

    @abstractmethod
    def grin_split(self, *args, **kwargs) -> Any:
        pass

    @staticmethod
    def check_dim(data: torch.Tensor):
        dim = data.ndim
        if dim == 3:
            return data
        elif data.ndim == 2:
            return rearrange(data, "s (n f) -> s n f", f=1)
        elif data.ndim == 1:
            return rearrange(data, "(s n f) -> s n f", n=1, f=1)
        else:
            raise ValueError(f"Invalid data dimensions {dim}")

    def expand_and_merge_indices(self, indices) -> np.ndarray:
        ds_indices = dict.fromkeys(
            [time for time in ["window", "horizon"] if getattr(self, time) > 0]
        )
        indices = np.arange(len(self._indices)) if indices is None else indices
        if "window" in ds_indices:
            window_idxs = [
                np.arange(idx, idx + self.window) for idx in self._indices[indices]
            ]
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
        ds_indices = np.unique(
            np.hstack([v for v in ds_indices.values() if v is not None])
        )
        return ds_indices
