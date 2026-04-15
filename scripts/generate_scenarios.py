#!/usr/bin/env python3
"""
Batch pre-generation of missingness scenarios using Hydra.

Usage:
    # Basic: generate for airq_default with default rates
    python scripts/generate_scenarios.py dataset.name=airq dataset.variant=default
    
    # Override rates via CLI
    python scripts/generate_scenarios.py \
        dataset.name=airq \
        'injection.target_rate=[0.25,0.30,0.35,0.40]' \
        injection.cumulative=false
    
    # Change pattern
    python scripts/generate_scenarios.py \
        dataset.name=airq \
        injection.pattern=mcar_points \
        injection.block_size=1
    
    # Force regeneration
    python scripts/generate_scenarios.py cache.force_regenerate=true
    
    # Multirun: generate for multiple datasets
    python scripts/generate_scenarios.py -m \
        dataset.name=airq \
        dataset.variant=default,small
    
    # Multirun: generate with different patterns
    python scripts/generate_scenarios.py -m \
        dataset.name=airq \
        injection.pattern=mcar_blocks,mcar_points
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from datasets.dataset_registry import DatasetRegistry
from datasets.missingness_scenarios import ScenarioManager
from datasets.missingness_scenarios._cache import ScenarioCache, compute_data_hash


def load_baseline_data(dataset_cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw data for scenario generation.

    Returns data and stations without any imputation or injection.
    """
    # Build loader config (minimal, no injection)
    loader_config = {
        "name": dataset_cfg.name,
        "variant": dataset_cfg.variant,
        "dataset_path": dataset_cfg.path,
        "small": dataset_cfg.get("small", False),
        "impute_nans": False,  # Keep original NaNs for baseline computation
    }

    # Get loader from registry
    loader = DatasetRegistry.get(OmegaConf.create(loader_config))

    # Load raw data
    data, stations, _ = loader.load_raw(small=dataset_cfg.get("small", False))

    return data, stations


def generate_and_cache_scenarios(
    data: pd.DataFrame,
    injection_cfg: DictConfig,
    cache_cfg: DictConfig,
) -> Dict:
    """
    Generate scenarios and save to cache.

    Returns index metadata for retrieval.
    """
    # Compute baseline missingness
    baseline_mask = (~np.isnan(data.values)).astype(int)
    baseline_missing_rate = 1.0 - baseline_mask.mean()

    # Initialize scenario manager
    mgr = ScenarioManager(cache_dir=cache_cfg.dir)
    mgr.set_data_hash(data.values)
    mgr.set_original_missing_from_mask(baseline_mask)

    # Prepare injection parameters
    target_rates = injection_cfg.target_rate
    if isinstance(target_rates, (int, float)):
        target_rates = [float(target_rates)]
    target_rates = sorted(target_rates)

    print("Generating scenarios:")
    print(f"Pattern: {injection_cfg.pattern}")
    print(f"Block size: {injection_cfg.block_size}")
    print(f"Target rates: {[f'{r:.0%}' for r in target_rates]}")
    print(f"Cumulative: {injection_cfg.cumulative}")
    print(f"Seed: {injection_cfg.seed}")
    print(f"Baseline missing: {baseline_missing_rate:.2%}")
    print(f"Cache dir: {cache_cfg.dir}")

    # Generate scenarios
    if injection_cfg.cumulative and len(target_rates) > 1:
        # Nested scenarios (cumulative ablation)
        scenarios = mgr.get_scenario_batch(
            shape=data.shape,
            base_missing_rate=baseline_missing_rate,
            target_rates=target_rates,
            pattern=injection_cfg.pattern,
            block_size=injection_cfg.block_size,
            seed=injection_cfg.seed,
            cumulative=True,
        )
    else:
        # Independent scenarios
        scenarios = {}
        for rate in target_rates:
            scenario = mgr.get_scenario(
                shape=data.shape,
                base_missing_rate=baseline_missing_rate,
                target_rate=rate,
                pattern=injection_cfg.pattern,
                block_size=injection_cfg.block_size,
                seed=injection_cfg.seed,
                force_regenerate=cache_cfg.force_regenerate,
            )
            scenarios[rate] = scenario

    # Build index metadata
    index = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "dataset": {
            "name": "airq",  # Could extract from config
            "variant": "default",
            "shape": list(data.shape),
            "hash": mgr._data_hash,
        },
        "baseline_missing_rate": float(baseline_missing_rate),
        "injection": {
            "pattern": injection_cfg.pattern,
            "block_size": injection_cfg.block_size,
            "seed": injection_cfg.seed,
            "cumulative": injection_cfg.cumulative,
        },
        "cache": {
            "dir": str(cache_cfg.dir),
            "force_regenerate": cache_cfg.force_regenerate,
        },
        "scenarios": {},
    }

    # Add scenario metadata to index
    for rate, scenario in scenarios.items():
        rate_str = str(rate)
        meta = scenario.metadata

        index["scenarios"][rate_str] = {
            "target_rate": float(rate),
            "actual_rate": float(
                meta.get("actual_rate", 1.0 - (scenario.full_mask == 1).mean())
            ),
            "eval_targets": int(scenario.eval_mask.sum()),
            "eval_fraction": float(scenario.eval_mask.mean()),
            "cache_key": scenario.config.get_cache_key(),
            "file": f"scenario_{scenario.config.get_cache_key()}.h5",
        }

        # Log progress
        print(
            f"   ✅ {rate_str}: {scenario.eval_mask.sum():,} eval targets "
            f"(actual: {meta.get('actual_rate', 'N/A'):.2%})"
        )

    return index


def save_index(index: Dict, output_cfg: DictConfig) -> Path:
    """Save index file and optionally backup."""
    output_path = Path(output_cfg.index_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n💾 Index saved to: {output_path}")

    # Optional backup
    if output_cfg.get("backup_dir"):
        backup_path = Path(output_cfg.backup_dir) / output_path.name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with open(backup_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"   📦 Backup saved to: {backup_path}")

    return output_path


def validate_index(index: Dict, cache_cfg: DictConfig) -> List[str]:
    """Validate generated index and return warnings."""
    warnings = []

    # Check dataset hash
    if cache_cfg.get("validate_hash", True) and not index.get("dataset", {}).get(
        "hash"
    ):
        warnings.append("⚠️  Dataset hash not recorded in index")

    # Check eval target counts
    min_eval = cache_cfg.get("min_eval_targets", 50000)
    for rate_str, meta in index.get("scenarios", {}).items():
        if meta.get("eval_targets", 0) < min_eval:
            warnings.append(
                f"⚠️  Rate {rate_str}: eval targets {meta['eval_targets']:,} "
                f"< minimum {min_eval:,}"
            )

    # Check for large undershoot
    for rate_str, meta in index.get("scenarios", {}).items():
        target = meta.get("target_rate")
        actual = meta.get("actual_rate")
        if target and actual and abs(actual - target) > 0.05:
            warnings.append(
                f"⚠️  Rate {rate_str}: target {target:.0%} → actual {actual:.2%} "
                f"(undershoot: {target - actual:.2%})"
            )

    return warnings


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

    print(f"   Shape: {data.shape}")
    print(f"   Columns: {list(data.columns)}")

    baseline_missing = np.isnan(data.values).mean()
    data_hash = compute_data_hash(data)

    print(f"   Shape: {data.shape}")
    print(f"   Baseline missing: {baseline_missing:.2%}")
    print(f"   Data hash: {data_hash}")

    # ========================================================================
    # STEP 2: Initialize Cache
    # ========================================================================

    cache = ScenarioCache(cache_dir=cfg.cache.dir)
    print(f"\n💾 Cache directory: {cfg.cache.dir}")

    # Check existing scenarios
    existing = cache.list_cached()
    if existing and not cfg.cache.force_regenerate:
        print(f"   Found {len(existing)} existing scenarios")
        available_rates = cache.list_available_rates()
        print(f"   Available rates: {[f'{r:.0%}' for r in available_rates]}")

    # ========================================================================
    # STEP 3: Generate Scenarios
    # ========================================================================

    baseline_mask = (~np.isnan(data.values)).astype(int)

    # Prepare target rates
    target_rates = cfg.injection.target_rate
    if isinstance(target_rates, (int, float)):
        target_rates = [float(target_rates)]
    target_rates = sorted(target_rates)

    print("\nGenerating scenarios:")
    print(f"   Pattern: {cfg.missingness.pattern}")
    print(f"   Block size: {cfg.missingness.block_size}")
    print(f"   Target rates: {[f'{r:.0%}' for r in target_rates]}")
    print(f"   Cumulative: {cfg.missingness.cumulative}")
    print(f"   Seed: {cfg.missingness.seed}")

    # Initialize scenario manager
    mgr = ScenarioManager(cache_dir=cfg.cache.dir)
    mgr.set_data_hash(data.values)
    mgr.set_original_missing_from_mask(baseline_mask)

    # Generate scenarios
    if cfg.injection.cumulative and len(target_rates) > 1:
        scenarios = mgr.get_scenario_batch(
            shape=data.shape,
            base_missing_rate=baseline_missing,
            target_rates=target_rates,
            pattern=cfg.missingness.pattern,
            block_size=cfg.missingness.block_size,
            seed=cfg.missingness.seed,
            cumulative=True,
        )
    else:
        scenarios = {}
        for rate in target_rates:
            scenario = mgr.get_scenario(
                shape=data.shape,
                base_missing_rate=baseline_missing,
                target_rate=target_rates[0],
                pattern=cfg.missingness.pattern,
                block_size=cfg.missingness.block_size,
                seed=cfg.missingness.seed,
                force_regenerate=cfg.cache.force_regenerate,
            )
            scenarios[rate] = scenario

    # ========================================================================
    # STEP 4: Summary
    # ========================================================================

    print("\n📋 Generation Summary:")
    for rate, scenario in scenarios.items():
        meta = scenario.metadata
        print(
            f"   {rate:.0%}: {scenario.eval_mask.sum():,} eval targets "
            f"(actual: {meta.get('actual_rate', 'N/A'):.2%})"
        )

    # List all cached scenarios
    all_cached = cache.list_cached()
    print(f"\n💾 Total cached scenarios: {len(all_cached)}")

    print("\n✅ Done!")

    return scenarios


if __name__ == "__main__":
    main()
