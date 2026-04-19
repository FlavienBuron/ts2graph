"""
Missingness Scenario Manager

Usage:
    from missingness_scenarios import ScenarioManager

    mgr = ScenarioManager(cache_dir="./cache")

    # Single scenario
    scenario = mgr.get_scenario(
        shape=(1000, 20),
        target_rate=0.40,
        pattern='mcar_blocks',
        block_size=10,
        seed=42,
        baseline_mask=original_missing,
    )

    # Batch with cumulative injection
    scenarios = mgr.get_scenario_batch(
        shape=(1000, 20),
        target_rates=[0.35, 0.40, 0.45],
        pattern='mcar_blocks',
        block_size=10,
        seed=42,
        baseline_mask=original_missing,
    )

    # Use in dataloader
    masked_data = data.copy()
    masked_data[scenario['full_mask'] == 0] = np.nan
    eval_targets = data[scenario['eval_mask'] == 1]
"""

import re
from typing import Dict, List, Optional

import numpy as np
from pandas.core.computation.ops import Op

from datasets.missingness_scenarios._cache import (
    ScenarioCache,
    compute_data_hash,
)
from datasets.missingness_scenarios._config import ScenarioConfig, ScenarioResult
from datasets.missingness_scenarios._generator import (
    MCARCumulativeGenerator,
    # inject_mar_systematic,
    inject_mcar_blocks,
    inject_mcar_points,
)


class ScenarioManager:
    """
    Generate and cache missingness scenarios for imputation studies.

    Convention: 1 = present (observed), 0 = missing
    """

    def __init__(
        self,
        cache_dir: str = "./missingness_cache",
    ):
        self.cache = ScenarioCache(cache_dir)
        self._original_missing_mask: Optional[np.ndarray] = None
        self._data_hash: Optional[str] = None

    def set_original_missing_from_mask(self, mask: np.ndarray):
        """
        Set original missingness mask from provided mask
        """
        self._original_missing_mask = mask.astype(bool)

    def set_data_hash(self, data: np.ndarray):
        """
        Compute and set data hash for cache validation.
        """
        self._data_hash = compute_data_hash(data)

    def get_scenario(
        self,
        shape: tuple,
        base_missing_rate: float,
        target_rate: float,
        pattern: str = "mcar_blocks",
        block_size: int = 10,
        seed: int = 42,
        force_regenerate: bool = False,
        eval_fraction: Optional[float] = None,
    ) -> ScenarioResult:
        """
        Get or generate a missingness scenario.

        Returns dict with:
            - 'full_mask': 1=present, 0=missing (baseline + injected)
            - 'eval_mask': 1=newly injected (evaluation targets)
            - 'baseline_mask': 1=present, 0=missing (original)
            - 'metadata': dict with actual_rate, eval_points, etc.
        """
        if self._original_missing_mask is None:
            self._original_missing_mask = np.ones(shape, dtype=int)

        config = ScenarioConfig(
            seed=seed,
            base_missing_rate=base_missing_rate,
            target_missing_rate=target_rate,
            pattern=pattern,
            block_size=block_size,
            dataset_shape=shape,
            dataset_hash=self._data_hash or "",
            injection_mode="independent",
            eval_fraction=eval_fraction,
            is_first_rate=eval_fraction is None,
        )

        # Try cache
        if not force_regenerate:
            cached = self.cache.load_scenario(config)
            if cached is not None:
                # Check if cached result matches requested eval_fraction
                cached_eval_fraction = cached.metadata.get("eval_fraction")
                if cached_eval_fraction is None:
                    raise AttributeError(
                        "Eval fraction could not be retrieved from the cached scenario"
                    )
                if eval_fraction is None or np.isclose(
                    cached_eval_fraction, eval_fraction
                ):
                    return cached

        # Generate new scenario
        print(f"Generating scenario: {config.get_cache_key()}")

        if pattern == "mcar_blocks":
            result = inject_mcar_blocks(
                config,
                self._original_missing_mask,
                eval_fraction,
            )
        elif pattern == "mcar_points":
            result = inject_mcar_points(
                config,
                self._original_missing_mask,
                eval_fraction,
            )
        # elif pattern == "mar_systematic":
        #     full_mask, eval_mask = inject_mar_systematic(
        #         base_missing_rate, target_rate, base_missing_rate, seed
        #     )
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        # Save cache
        self.cache.save(result)

        return result

    def get_scenario_batch(
        self,
        shape: tuple,
        base_missing_rate: float,
        target_rates: List[float],
        pattern: str = "mcar_blocks",
        block_size: int = 10,
        seed: int = 42,
        cumulative: bool = True,
        force_regenerate: bool = False,
    ) -> Dict[float, ScenarioResult]:
        """
        Get multiple scenarios. If cumulative=True, they are nested.
        """
        if self._original_missing_mask is None:
            self._original_missing_mask = np.ones(shape, dtype=int)

        results = {}
        T, N = self._original_missing_mask.shape
        total_positions = T * N

        if cumulative and pattern == "mcar_blocks":
            generator = MCARCumulativeGenerator(
                baseline_mask=self._original_missing_mask, seed=seed
            )

            for i, rate in enumerate(sorted(target_rates)):
                config = ScenarioConfig(
                    seed=seed,
                    base_missing_rate=base_missing_rate,
                    target_missing_rate=rate,
                    pattern=pattern,
                    block_size=block_size,
                    dataset_shape=shape,
                    dataset_hash=self._data_hash or "",
                    injection_mode="cumulative",
                    eval_fraction=None,
                    is_first_rate=(i == 0),
                )
                # Try cache
                if not force_regenerate:
                    cached = self.cache.load_scenario(config)
                    if cached is not None:
                        generator.cumulative_eval_mask = cached.eval_mask_cumulative
                        if config.is_first_rate:
                            generator.fixed_eval_mask = cached.eval_mask_fixed
                            generator.first_rate = rate
                            generator.first_rate_eval_fraction = cached.metadata[
                                "eval_fraction"
                            ]
                        results[rate] = cached
                        continue

                result = generator.inject_block_to_rate(config)

                result.metadata.update(
                    {
                        "cumulative": True,
                    }
                )

                self.cache.save(result)

                results[rate] = result
        else:
            # ================================================================
            # INDEPENDENT MODE: Subsample eval masks to first rate's fraction
            # ================================================================

            # Step 1: Generate first rate to determine eval fraction
            first_rate = sorted(target_rates)[0]
            first_result = self.get_scenario(
                shape=shape,
                target_rate=first_rate,
                pattern=pattern,
                block_size=block_size,
                seed=seed + int(first_rate * 100),
                base_missing_rate=base_missing_rate,
                eval_fraction=None,
            )
            eval_fraction = float(first_result.eval_mask_fixed.mean())
            self.cache.save(first_result)
            results[first_rate] = first_result

            for rate in target_rates[1:]:
                result = self.get_scenario(
                    shape=shape,
                    target_rate=rate,
                    pattern=pattern,
                    block_size=block_size,
                    seed=seed + int(rate * 100),
                    base_missing_rate=base_missing_rate,
                    eval_fraction=eval_fraction,
                )
                self.cache.save(result)
                results[rate] = result

        return results

    def clear_cache(self) -> None:
        """Clear all cached scenarios."""
        self.cache.clear()
