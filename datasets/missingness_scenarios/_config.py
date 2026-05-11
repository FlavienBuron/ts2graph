import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class ScenarioConfig:
    """
    Immutable configuration for a missingness scenario.
    Used as cache key: any change invalidates cached masks
    """

    seed: int
    base_missing_rate: float
    target_missing_rate: float
    pattern: str
    block_size: int = 10
    min_sensors_covered: float = 0.9
    dataset_shape: tuple[int, int] = field(default=(0, 0))
    dataset_hash: str = field(default="")
    injection_mode: str = field(default="independent")  # "cumulative" or "independent"

    # Eval fraction for subsampling (affects eval_mask_fixed)
    # For cumulative: None (no subsampling, first rate determines fraction)
    # For independent: float (subsample to this fraction)
    eval_fraction: Optional[float] = field(default=None)

    # For cumulative mode, track if this is the first rate
    # (first rate determines fixed eval mask)
    is_first_rate: bool = field(default=False)

    # Aligned/Block-specific parameters
    sensor_fraction: float = 0.2
    sensor_pattern: str = "top"  # top, random,
    placement: str = "span_all"  # span_all, test_only, train_only, random_quarter
    test_months: list[int] = field(default_factory=lambda: [3, 6, 9, 12])
    aligned_across_sensors: bool = field(default=False)

    def to_dict(self) -> dict:
        """Convert to dict, ensuring all values are JSON-serializable."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()
        float_fields = ["base_missing_rate", "target_missing_rate", "eval_fraction", "min_sensors_covered", "sensor_fraction"]
        for k in float_fields:
            if isinstance(d.get(k), float):
                d[k] = round(d[k], 6)

        if isinstance(d.get("dataset_shape"), tuple):
            d["dataset_shape"] = list(d["dataset_shape"])
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def get_cache_key(self) -> str:
        """
        Generate unique cache key from config
        """
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioConfig":
        """Create from dict."""
        d = d.copy()
        if "dataset_shape" in d and isinstance(d["dataset_shape"], list):
            d["dataset_shape"] = tuple(d["dataset_shape"])
        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> "ScenarioConfig":
        return cls.from_dict(json.loads(json_str))


@dataclass
class ScenarioResult:
    """
    Output from scenario generation
    """

    config: ScenarioConfig
    baseline_mask: np.ndarray
    full_mask: np.ndarray  # baseline missing data + all injections (previous + new)

    # Eval mask options
    eval_mask_fixed: np.ndarray  # missing data injected in first rate
    eval_mask_newly: np.ndarray  # Injected at this rate only
    eval_mask_cumulative: np.ndarray  # All eval positions up to current rate

    metadata: dict = field(default_factory=dict)

    def __verify__(self):
        """Validate mask shapes and relationships."""
        assert self.full_mask.shape == self.baseline_mask.shape
        assert self.full_mask.shape == self.eval_mask_fixed.shape
        assert self.full_mask.shape == self.eval_mask_newly.shape
        assert self.full_mask.shape == self.eval_mask_cumulative.shape

        # Verify cumulative = fixed OR newly (no overlap)
        expected_cumulative = np.logical_or(self.eval_mask_fixed, self.eval_mask_newly)
        assert np.array_equal(self.eval_mask_cumulative, expected_cumulative), "eval_mask_cumulative should be logical OR of fixed and newly"

    def save_hdf5(self, filepath: str):
        """
        Save masks and metadata to HDF5 file
        """
        import h5py

        with h5py.File(filepath, "w") as f:
            # Store masks as datasets
            f.create_dataset("full_mask", data=self.full_mask.tolist())
            f.create_dataset("baseline_mask", data=self.baseline_mask.tolist())

            f.create_dataset("eval_mask_fixed", data=self.eval_mask_fixed.tolist())
            f.create_dataset("eval_mask_newly", data=self.eval_mask_newly.tolist())
            f.create_dataset("eval_mask_cumulative", data=self.eval_mask_cumulative.tolist())

            # Store config as JSON string attribute
            f.attrs["config_json"] = self.config.to_json()
            f.attrs["cache_key"] = self.config.get_cache_key()

            # Store metadata as JSON string attribute
            f.attrs["metadata_json"] = json.dumps(self.metadata)

            # Store creation info
            import datetime

            f.attrs["created_at"] = datetime.datetime.now().isoformat()

    @classmethod
    def load_hdf5(cls, filepath: str) -> "ScenarioResult":
        """
        Load masks and metadata from HDF5 file
        """
        from typing import cast

        import h5py

        with h5py.File(filepath, "r") as f:
            # Load masks
            full_mask = cast(np.ndarray, f["full_mask"])[:]
            baseline_mask = cast(np.ndarray, f["baseline_mask"])[:]
            eval_mask_fixed = cast(np.ndarray, f["eval_mask_fixed"])[:]
            eval_mask_newly = cast(np.ndarray, f["eval_mask_newly"])[:]
            eval_mask_cumulative = cast(np.ndarray, f["eval_mask_cumulative"])[:]

            # Load config from attributes
            config = ScenarioConfig.from_json(str(f.attrs["config_json"]))
            metadata = json.loads(str(f.attrs["metadata_json"]))

            return cls(
                config=config,
                full_mask=full_mask.astype(bool),
                baseline_mask=baseline_mask.astype(bool),
                eval_mask_fixed=eval_mask_fixed.astype(int),
                eval_mask_newly=eval_mask_newly.astype(int),
                eval_mask_cumulative=eval_mask_cumulative.astype(int),
                metadata=metadata,
            )

    def get_eval_mask(self, mode: str = "fixed") -> np.ndarray:
        """
        Get eval mask by mode for evaluation.

        Parameters
        ----------
        mode : str
            "fixed" → Use first rate eval positions (clean ablation)
            "newly" → Use only this rate's eval positions
            "cumulative" → Use all eval positions up to this rate

        Returns
        -------
        eval_mask : np.ndarray
            Boolean mask (True = eval target)
        """
        if mode == "fixed":
            return self.eval_mask_fixed
        elif mode == "newly":
            return self.eval_mask_newly
        elif mode == "cumulative":
            return self.eval_mask_cumulative
        else:
            raise ValueError(f"Unknown eval mask mode: {mode}")

    def get_eval_mask_stats(self) -> Dict[str, int]:
        """Get statistics for all eval mask types."""
        return {
            "eval_fixed": int(self.eval_mask_fixed.sum()),
            "eval_newly": int(self.eval_mask_newly.sum()),
            "eval_cumulative": int(self.eval_mask_cumulative.sum()),
        }
