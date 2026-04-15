import hashlib
import json
from dataclasses import asdict, dataclass, field

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

    def to_dict(self) -> dict:
        """Convert to dict, ensuring all values are JSON-serializable."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()
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
    full_mask: np.ndarray
    eval_mask: np.ndarray
    baseline_mask: np.ndarray
    metadata: dict = field(default_factory=dict)

    def save_hdf5(self, filepath: str):
        """
        Save masks and metadata to HDF5 file
        """
        import h5py

        with h5py.File(filepath, "w") as f:
            # Store masks as datasets
            f.create_dataset("full_mask", data=self.full_mask.tolist())
            f.create_dataset("eval_mask", data=self.eval_mask.tolist())
            f.create_dataset("baseline_mask", data=self.baseline_mask.tolist())

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
            eval_mask = cast(np.ndarray, f["eval_mask"])[:]
            baseline_mask = cast(np.ndarray, f["baseline_mask"])[:]

            # Load config from attributes
            config = ScenarioConfig.from_json(str(f.attrs["config_json"]))
            metadata = json.loads(str(f.attrs["metadata_json"]))

            return cls(
                config=config,
                full_mask=full_mask.astype(bool),
                eval_mask=eval_mask.astype(int),
                baseline_mask=baseline_mask.astype(bool),
                metadata=metadata,
            )
