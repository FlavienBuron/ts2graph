#!/usr/bin/env python3
"""
Batch pre-generation of missingness scenarios using Hydra.

Usage:
    # Basic: generate for airq_default with default rates
    python scripts/generate_scenarios.py dataset.name=airq dataset.variant=default
    
    # Override rates via CLI
    python scripts/generate_scenarios.py \
        dataset.name=airq \
        'missingness.target_rates=[0.25,0.30,0.35,0.40]' \
        missingness.cumulative=false
    
    # Change pattern
    python scripts/generate_scenarios.py \
        dataset.name=airq \
        missingness.pattern=mcar_points \
        missingness.block_size=1
    
    # Force regeneration
    python scripts/generate_scenarios.py cache.force_regenerate=true
    
    # Multirun: generate for multiple datasets
    python scripts/generate_scenarios.py -m \
        dataset.name=airq \
        dataset.variant=default,small
    
    # Multirun: generate with different patterns
    python scripts/generate_scenarios.py -m \
        dataset.name=airq \
        missingness.pattern=mcar_blocks,mcar_points
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

import datasets.dataloaders  # noqa: F401
from datasets.dataset_registry import DatasetRegistry
from datasets.missingness_scenarios import ScenarioManager
from datasets.missingness_scenarios._cache import compute_data_hash


def load_baseline_data(dataset_cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw data for scenario generation.

    Returns data and stations without any imputation or injection.
    """

    # Get loader from registry
    loader = DatasetRegistry.get(dataset_cfg)

    # Load raw data
    data, stations, _ = loader.load_raw(small=dataset_cfg.get("small", False))

    return data, stations


@hydra.main(config_path="../configs", config_name="missingness", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point for scenario generation.

    Hydra config is automatically loaded and can be overridden via CLI.
    """
    # Print header
    print("\n" + "=" * 60)
    print("MISSINGNESS SCENARIO GENERATION")
    print("=" * 60)

    # ========================================================================
    # STEP 1: Load Baseline Data
    # ========================================================================

    print(f"\n📦 Loading baseline  {cfg.dataset.name}_{cfg.dataset.variant}")
    data, stations = load_baseline_data(cfg.dataset)
    baseline_mask = (~np.isnan(data.values)).astype(int)

    print(f"   Shape: {data.shape}")

    baseline_missing = np.round(np.isnan(data.values).mean(), 6)
    data_hash = compute_data_hash(data.values)

    print(f"   Shape: {data.shape}")
    print(f"   Baseline missing: {baseline_missing:.2%}")
    print(f"   Data hash: {data_hash}")

    # ========================================================================
    # STEP 2: Initialize Cache
    # ========================================================================

    # Initialize scenario manager
    mgr = ScenarioManager(cache_dir=cfg.cache.dir)
    mgr.set_data_hash(data.values)
    mgr.set_original_missing_from_mask(baseline_mask)

    print(f"\nCache directory: {cfg.cache.dir}")

    # Check existing scenarios
    existing = mgr.cache.list_cached()
    if existing and not cfg.cache.force_regenerate:
        print(f"   Found {len(existing)} existing scenarios")
        available_rates = mgr.cache.list_available_rates()
        print(f"   Available rates: {[f'{r:.0%}' for r in available_rates]}")

    # ========================================================================
    # STEP 3: Generate Scenarios
    # ========================================================================

    pattern = cfg.missingness.pattern
    is_aligned = pattern in ("aligned_blocks", "aligned")

    # Determine target levels (rates for MCAR, sensor fractions for aligned)
    if is_aligned:
        target_rates = sorted([float(x) for x in cfg.missingness.sensor_coverage_levels])
        print(f"   Pattern: {pattern} (sensor coverage levels)")
    else:
        target_rates = sorted([float(r) for r in cfg.missingness.target_rates])
        print(f"   Pattern: {pattern} (global missing rates)")

    print("\nGenerating scenarios:")
    print(f"   Pattern: {cfg.missingness.pattern}")
    print(f"   Block size: {cfg.missingness.block_size}")
    print(f"   Target rates: {[f'{r:.0%}' for r in target_rates]}")
    print(f"   Cumulative: {cfg.missingness.cumulative}")
    if is_aligned:
        print(f"   Eval fraction (upper bound): {cfg.missingness.eval_fraction:.1%}")
    print(f"   Seed: {cfg.missingness.seed}")

    # Extract data_index safely for aligned patterns
    data_index = pd.DatetimeIndex(data.index) if is_aligned else None

    # Generate scenarios
    # if cfg.missingness.cumulative and len(target_rates) > 1:
    scenarios = mgr.get_scenario_batch(
        shape=data.shape,
        base_missing_rate=baseline_missing,
        target_rates=target_rates,
        pattern=pattern,
        block_size=cfg.missingness.block_size,
        seed=cfg.missingness.seed,
        cumulative=cfg.missingness.cumulative,
        force_regenerate=cfg.cache.force_regenerate,
        eval_fraction=cfg.missingness.eval_fraction if is_aligned else None,
        # Aligned kwargs (ignored for MCAR)
        data_index=data_index,
        sensor_pattern=cfg.missingness.aligned.sensor_pattern,
        placement=cfg.missingness.aligned.placement,
        test_months=list(cfg.missingness.aligned.test_months),
    )
    # else:
    #     scenarios = mgr.get_scenario_batch(
    #         shape=data.shape,
    #         base_missing_rate=baseline_missing,
    #         target_rates=target_rates,
    #         pattern=cfg.missingness.pattern,
    #         block_size=cfg.missingness.block_size,
    #         seed=cfg.missingness.seed,
    #         cumulative=False,  # ← Independent mode
    #         force_regenerate=cfg.cache.force_regenerate,
    #     )

    # ========================================================================
    # STEP 4: Summary
    # ========================================================================

    print("\n📋 Generation Summary:")
    for rate, scenario in scenarios.items():
        meta = scenario.metadata
        actual_rate = meta.get("actual_rate", "N/A")
        eval_pts = scenario.eval_mask_fixed.sum()

        if is_aligned:
            coverage = meta.get("achieved_sensor_coverage", "N/A")
            eval_frac = meta.get("eval_fraction_achieved", 0)
            print(f"   {rate:.0%} coverage: {eval_pts:,} eval targets (achieved coverage: {coverage:.0%}, eval: {eval_frac:.1%})")
        else:
            print(f"   {rate:.0%} rate: {eval_pts:,} eval targets (actual missing: {actual_rate:.2%})")

    # List all cached scenarios
    all_cached = mgr.cache.list_cached()
    print(f"\n💾 Total cached scenarios: {len(all_cached)}")

    print("\n✅ Done!")

    return scenarios


if __name__ == "__main__":
    main()
