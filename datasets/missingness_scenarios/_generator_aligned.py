from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from datasets.missingness_scenarios._config import ScenarioConfig, ScenarioResult


class AlignedBlockCumulativeGenerator:
    """
    Injects aligned mono-blocks of missing data cumulatively.

    Same temporal block position/size across affected sensors.
    Each missingness level builds on the previous one.

    Simulates: Hub failures, regional outages, synchronized sensor issues.

    Uses only ScenarioConfig/ScenarioResult (no custom state classes).
    """

    def __init__(self, baseline_mask: np.ndarray, data_index: pd.DatetimeIndex, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.baseline_mask = baseline_mask.astype(int)
        self.data_index = data_index
        self.T, self.N = baseline_mask.shape
        self.total_elements = self.T * self.N

        # Cumulative state (instance variables, not custom dataclass)
        self._current_mask = self.baseline_mask.copy()
        self._cumulative_eval_mask = np.zeros_like(baseline_mask, dtype=bool)
        self._fixed_eval_mask: Optional[np.ndarray] = None
        self._first_rate: Optional[float] = None
        self._first_rate_eval_fraction: float = 0.0
        self._affected_sensors: Set[int] = set()
        self._last_block_end: Optional[int] = None

    def inject_to_rate(
        self,
        config: ScenarioConfig,
    ) -> ScenarioResult:
        """
        Inject an aligned block at target missing rate, updating cumulative state.

        Parameters mirror the stateless function for API consistency.
        Returns ScenarioResult using only defined classes.
        """
        if config.sensor_fraction <= 0:
            raise ValueError(f"sensor_fraction must be > 0 for aligned block pettern, got {config.sensor_fraction}")

        # Compute block placement (single position for all affected sensors)
        block_start, block_end = _compute_aligned_placement(
            block_size=config.block_size,
            data_index=self.data_index,
            placement=config.placement,
            test_months=config.test_months,
            rng=self.rng,
        )
        block_end = min(block_end, self.T)

        target_total = int(self.N * config.sensor_fraction)
        current_total = len(self._affected_sensors)
        n_new = max(0, target_total - current_total)

        new_sensors = _select_aligned_sensors(
            n_new=n_new,
            existing_sensors=self._affected_sensors,
            pattern=config.sensor_pattern,
            N=self.N,
            rng=self.rng,
        )
        self._affected_sensors.update(new_sensors)
        affected_sensors = list(self._affected_sensors)

        # Create newly injected mask (only new sensors × current block window)
        block_mask = np.zeros_like(self._current_mask, dtype=bool)
        block_mask[block_start:block_end, list(affected_sensors)] = True
        newly_injected = block_mask & (self.baseline_mask == 1)
        self._current_mask[block_mask] = 0
        newly_injected &= self.baseline_mask == 1

        # Unmark previously-injected sensors in prior time windows (for cumulative "both" mode)
        if self._last_block_end is not None:
            prev_sensors = self._affected_sensors - new_sensors
            if prev_sensors:
                newly_injected[: self._last_block_end, list(prev_sensors)] = False

        # Apply injection to current mask
        self._current_mask[block_start:block_end, affected_sensors] = 0

        # Update cumulative eval mask
        self._cumulative_eval_mask = np.logical_or(self._cumulative_eval_mask, newly_injected)

        # if config.eval_fraction is not None and newly_injected.any():
        #     subsample_rng = np.random.default_rng(config.seed + hash(config.target_missing_rate) % 10000)
        #     fixed_eval_mask = _subsample_eval_mask(
        #         newly_injected=newly_injected,
        #         target_eval_fraction=config.eval_fraction,
        #         total_positions=self.total_elements,
        #         block_start=block_start,
        #         block_end=block_end,
        #         affected_sensors=affected_sensors,
        #         rng=subsample_rng,
        #     )
        #     eval_mask_types = ["fixed_subsampled", "newly_full", "cumulative"]
        # else:
        #     raise ValueError("eval_fraction should be defined for Aligned Blocks case")

        # Set fixed eval mask at first successful injection
        if self._fixed_eval_mask is None:
            self._fixed_eval_mask: np.ndarray = newly_injected.copy()
            self._first_rate = config.sensor_fraction
            self._first_rate_eval_fraction = float(newly_injected.mean())
            print(f"Fixed eval mask set at {config.sensor_fraction:.0%}: {self._fixed_eval_mask.sum():,} positions ({self._first_rate_eval_fraction:.1%})")

        self._last_block_end = block_end

        blocks_injected = [[int(block_start), int(block_end), int(s)] for s in new_sensors]
        achieved_coverage = len(self._affected_sensors) / self.N
        achieved_missing_rate = float(1.0 - self._current_mask.mean())

        return ScenarioResult(
            config=config,
            full_mask=self._current_mask.copy(),
            baseline_mask=self.baseline_mask.copy(),
            eval_mask_fixed=self._fixed_eval_mask.copy(),
            eval_mask_newly=newly_injected.copy(),
            eval_mask_cumulative=self._cumulative_eval_mask.copy(),
            metadata={
                "target_sensor_fraction": float(config.sensor_fraction),
                "achieved_sensor_coverage": float(achieved_coverage),
                "blocks_count": len(blocks_injected),
                "blocks_injected": blocks_injected,
                "newly_injected_count": int(newly_injected.sum()),
                "actual_rate": float(achieved_missing_rate),
                "injection_mode": "cumulative",
                "eval_mask_types": ["fixed", "newly", "cumulative"],
                "eval_fraction": float(self._fixed_eval_mask.sum() / self.total_elements),
                "eval_fraction_requested": float(self._first_rate_eval_fraction),
                "eval_fixed_points": int(self._fixed_eval_mask.sum()),
                "eval_newly_points": int(newly_injected.sum()),
                "eval_cumulative_points": int(self._cumulative_eval_mask.sum()),
            },
        )


def inject_aligned_blocks(
    config: ScenarioConfig,
    baseline_mask: np.ndarray,
    data_index: pd.DatetimeIndex,
) -> ScenarioResult:
    """
    Inject aligned mono-blocks (stateless, independent scenario).

    Same temporal block position/size across affected sensors.
    Does not maintain state between calls.

    Uses only ScenarioConfig/ScenarioResult (no custom state classes).
    """
    rng = np.random.default_rng(config.seed)
    T, N = baseline_mask.shape
    total_elements = T * N

    current_mask = baseline_mask.copy()

    if config.sensor_fraction <= 0:
        raise ValueError(f"sensor_fraction must be > 0 for aligned block pettern, got {config.sensor_fraction}")

    # Compute single aligned block position
    block_start, block_end = _compute_aligned_placement(
        block_size=config.block_size,
        data_index=data_index,
        placement=config.placement,
        test_months=config.test_months,
        rng=rng,
    )
    block_end = min(block_end, T)

    # Select sensors (non-cumulative for independent mode)
    affected_sensors = _select_aligned_sensors(
        n_new=int(N * config.sensor_fraction),
        existing_sensors=set(),
        pattern=config.sensor_pattern,
        N=N,
        rng=rng,
    )

    # Apply injection# 1. Create full block mask (all positions in block for affected sensors)
    block_mask = np.zeros_like(current_mask, dtype=bool)
    block_mask[block_start:block_end, list(affected_sensors)] = True
    newly_injected = block_mask & (baseline_mask == 1)
    current_mask[block_mask] = 0
    newly_injected &= baseline_mask == 1

    if config.eval_fraction is not None and newly_injected.any():
        subsample_rng = np.random.default_rng(config.seed + hash(config.target_missing_rate) % 10000)
        eval_mask_fixed = _subsample_eval_mask(
            newly_injected=newly_injected,
            target_eval_fraction=config.eval_fraction,
            total_positions=total_elements,
            block_start=block_start,
            block_end=block_end,
            affected_sensors=list(affected_sensors),
            rng=subsample_rng,
        )
        eval_mask_types = ["fixed_subsampled", "newly_full"]
    else:
        # raise ValueError("eval_fraction should be defined for Aligned Blocks case")
        eval_mask_fixed = newly_injected
        eval_mask_types = ["rate_specific"]

    blocks_injected = [[int(block_start), int(block_end), int(s)] for s in affected_sensors]
    achieved_coverage = len(affected_sensors) / N
    achieved_missing_rate = float(1.0 - current_mask.mean())

    return ScenarioResult(
        config=config,
        full_mask=current_mask.copy(),
        baseline_mask=baseline_mask.copy(),
        eval_mask_fixed=eval_mask_fixed.copy(),
        eval_mask_newly=newly_injected.copy(),
        eval_mask_cumulative=newly_injected.copy(),
        metadata={
            "target_sensor_fraction": float(config.sensor_fraction),
            "achieved_sensor_coverage": float(achieved_coverage),
            "blocks_count": len(blocks_injected),
            "blocks_injected": blocks_injected,
            "newly_injected_count": int(newly_injected.sum()),
            "actual_rate": float(achieved_missing_rate),
            "injection_mode": "independent",
            "eval_fraction": float(eval_mask_fixed.sum() / total_elements),
            "eval_fraction_requested": float(config.eval_fraction or 0),
            "eval_mask_types": eval_mask_types,
            "eval_fixed_points": int(eval_mask_fixed.sum()),
            "eval_newly_points": int(newly_injected.sum()),
        },
    )


def _compute_aligned_placement(
    block_size: int,
    data_index: pd.DatetimeIndex,
    placement: str,
    test_months: List[int],
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """Compute aligned block start/end respecting placement strategy."""
    T = len(data_index)
    if block_size >= T:
        return 0, T

    def find_quarter(test_month: int) -> Optional[Tuple[int, int, int]]:
        month_mask = data_index.to_series().dt.month == test_month
        if not month_mask.any():
            return None
        test_idx = np.where(month_mask)[0]
        test_start, test_end = test_idx.min(), test_idx.max() + 1
        prev = [(test_month - 2) % 12 or 12, (test_month - 1) % 12 or 12]
        prev_mask = data_index.to_series().dt.month.isin(prev)
        q_start = np.where(prev_mask)[0].min() if prev_mask.any() else 0
        return int(q_start), int(test_start), int(test_end)

    valid_quarters = [q for tm in test_months if (q := find_quarter(tm)) is not None]
    if not valid_quarters:
        quarter_start, test_start, test_end = 0, 0, T
    else:
        quarter_start, test_start, test_end = rng.choice(valid_quarters)
        quarter_start, test_start, test_end = int(quarter_start), int(test_start), int(test_end)

    # 🔒 Helper: clamp start to [0, T - block_size] to preserve exact length
    def clamp_start(start: int) -> Tuple[int, int]:
        start = max(0, min(start, T - block_size))
        return int(start), int(start + block_size)

    if placement == "span_all":
        # Goal: block must overlap BOTH pre-test (quarter) AND test regions
        # Valid start range ensures ≥1 timestep in each region
        min_start = max(0, test_start - block_size + 1)  # Ensures end > test_start
        max_start = min(T - block_size, test_end - 1)  # Ensures start < test_end

        if min_start > max_start:
            # Window too narrow for this block size → fallback to random valid placement
            start = int(rng.integers(0, T - block_size + 1))
        else:
            # Random placement within valid overlap range
            start = int(rng.integers(min_start, max_start + 1))
        return clamp_start(start)

    elif placement == "test_only":
        # Center block on test window, allow overshoot into train/val or post-test
        center = (test_start + test_end) // 2
        start = center - block_size // 2
        return clamp_start(start)

    elif placement == "train_only":
        # Center block on pre-test (quarter) window
        center = quarter_start + (test_start - quarter_start) // 2
        start = center - block_size // 2
        return clamp_start(start)

    elif placement == "random_quarter":
        # Random placement anywhere within the full quarter span
        q_span = test_end - quarter_start
        max_start = max(quarter_start, T - block_size)
        start = int(rng.integers(quarter_start, min(max_start, T - block_size) + 1))
        return clamp_start(start)

    else:
        raise ValueError(f"Unknown placement: {placement}")


def _select_aligned_sensors(
    n_new: int,
    existing_sensors: Set[int],
    pattern: str,
    N: int,
    rng: np.random.Generator,
) -> Set[int]:
    """Select sensors according to pattern, excluding already-affected."""
    available = [s for s in range(N) if s not in existing_sensors]
    if not available or n_new <= 0:
        return set()

    n_select = min(n_new, len(available))

    if pattern == "top":
        selected = [s for s in sorted(available)][:n_select]
    elif pattern == "random":
        selected = list(rng.choice(available, size=n_select, replace=False))
    else:
        raise ValueError(f"Unknown sensor_pattern: {pattern}")

    return set(selected)


def _subsample_eval_mask(
    newly_injected: np.ndarray,
    target_eval_fraction: float,  # ← Fraction of TOTAL dataset (T × N)
    total_positions: int,  # ← T * N
    block_start: int,
    block_end: int,
    affected_sensors: List[int],
    rng: np.random.Generator,
) -> np.ndarray:
    # 1. Target budget relative to TOTAL dataset
    target_count = int(target_eval_fraction * total_positions)

    # 2. Cap at available injected points (strict subset constraint)
    available_count = int(newly_injected.sum())
    actual_count = min(target_count, available_count)

    if actual_count <= 0:
        return np.zeros_like(newly_injected, dtype=bool)

    # 3. If budget covers or exceeds block → use ENTIRE aligned block
    if actual_count >= available_count:
        return newly_injected.copy()

    # 4. Otherwise: subsample to budget while preserving aligned structure
    n_sensors = len(affected_sensors)
    block_len = block_end - block_start
    n_timesteps = max(1, min(actual_count // n_sensors, block_len))

    max_start = block_end - n_timesteps
    start = int(rng.integers(block_start, max_start + 1)) if max_start >= block_start else block_start

    eval_mask = np.zeros_like(newly_injected, dtype=bool)
    eval_mask[start : start + n_timesteps, affected_sensors] = True
    eval_mask &= newly_injected
    return eval_mask
