"""
HDF5 caching utilities (internal module).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from datasets.missingness_scenarios._config import ScenarioConfig, ScenarioResult


class ScenarioCache:
    """
    Manages cached scenario masks in HDF5 format
    """

    def __init__(
        self, cache_dir: str = "./datasets/missingness_scenarios/cache"
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, config: ScenarioConfig) -> Path:
        """
        Get HDF5 cache file path for config
        """
        cache_key = config.get_cache_key()
        return self.cache_dir / f"scenario_{cache_key}.h5"

    def exists(self, config: ScenarioConfig) -> bool:
        """
        Check if valid cached scenario exists
        """
        return self._get_cache_path(config).exists()

    def load_scenario(self, config: ScenarioConfig) -> Optional[ScenarioResult]:
        """
        Load cached scenario if valid
        """
        if not self.exists(config):
            return None

        cache_path = self._get_cache_path(config)
        try:
            result = ScenarioResult.load_hdf5(str(cache_path))

            # Validate config matches
            if result.config.get_cache_key() != config.get_cache_key():
                print(f"!!! Cache key mismatch for {config.get_cache_key()} !!!")
                return None

            # Validate dataset hash if provided
            if config.dataset_hash and result.config.dataset_hash:
                if config.dataset_hash != result.config.dataset_hash:
                    print("!!! Dataset hash mismatch. Data may have changed !!!")
                    return None

            print(f"Loaded cached scenario: {config.get_cache_key()}")
            print(f"    File: {cache_path.name}")
            return result
        except Exception as e:
            print(f"!!! Cache load failed. Exception: {e} !!!")
            return None

    def save(self, result: ScenarioResult) -> None:
        """
        Save scenario to HDF5 case
        """
        cache_path = self._get_cache_path(result.config)

        result.save_hdf5(str(cache_path))

        file_size = cache_path.stat().st_size
        print(f"Save scenario: {result.config.get_cache_key()}")
        print(f"    File: {cache_path.name} ({self._format_size(file_size)})")

    def clear(self):
        """
        Clear all cached scenarios.
        """
        removed = 0
        for f in self.cache_dir.glob("*.h5"):
            f.unlink()
            removed += 1
        print(
            f"Cleared {removed} cached scenario{'s' if removed > 1 else ''} from {self.cache_dir}"
        )

    def list_cached(self) -> List[Dict]:
        """
        List all cached scenario config with metadata.
        """
        import h5py

        cached = []

        for h5_path in self.cache_dir.glob("*.h5"):
            try:
                with h5py.File(str(h5_path), "r") as f:
                    config_json = str(f.attrs.get("config_json", "{}"))
                    metadata_json = str(f.attrs.get("metadata_json", "{}"))
                    created_at = str(f.attrs.get("created_at", "unknown"))
                    file_size = h5_path.stat().st_size

                    cached.append(
                        {
                            "cache_key": str(f.attrs.get("cache_key", "unknown")),
                            "filename": h5_path.name,
                            "config": json.loads(config_json) if config_json else {},
                            "metadata": json.loads(metadata_json)
                            if metadata_json
                            else {},
                            "created_at": created_at,
                            "file_size": file_size,
                            "file_size_formatted": self._format_size(file_size),
                        }
                    )
            except Exception as e:
                print(f"!!! Failed to read {h5_path.name}. Exception: {e} !!!")

        return cached

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.
        """
        h5_files = list(self.cache_dir.glob("*.h5"))
        total_size = sum(f.stat().st_size for f in h5_files)

        return {
            "num_scenarios": len(h5_files),
            "total_size": total_size,
            "total_size_formatted": self._format_size(total_size),
            "cache_dir": str(self.cache_dir),
        }

    def _format_size(self, size_bytes: float):
        """
        Format file size in human-readable format
        """

        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def list_available_rates(
        self, pattern: Optional[str] = None, block_size: Optional[int] = None
    ) -> List[float]:
        """List all available target rates in cache."""
        cached = self.list_cached()
        rates = []

        for entry in cached:
            config = entry.get("config", {})
            if pattern and config.get("pattern") != pattern:
                continue
            if block_size and config.get("block_size") != block_size:
                continue
            rate = config.get("target_missing_rate")
            if rate is not None:
                rates.append(float(rate))

        return sorted(set(rates))


def compute_cache_key(params: Dict[str, Any]) -> str:
    """
    Generate cache key from parameters
    """
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def compute_data_hash(data: Union[np.ndarray, pd.DataFrame]) -> str:
    """
    Compute hash of dataset for cache validation
    """
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]
