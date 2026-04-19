from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from datasets.missingness_scenarios._config import ScenarioConfig, ScenarioResult


@dataclass
class InjectionResult:
    "Track injection state and evaluation targets for one missingness level"

    mask: np.ndarray  # Full mask: original + injected
    newly_injected: np.ndarray  # Boolean: positions injected at current level
    target_rate: float  # Target total withheld rate
    actual_rate: float  # Actual achieved rate
    blocks_injected: List[Tuple[int, int, int]]
    metadata: Dict = field(default_factory=dict)


class MCARCumulativeGenerator:
    """
    Injects MCAR blocks of missing data, traking them as an evaluatation mask
    Each missingness level build on the previous one (35% ⊂ 40% ⊂ 45%)
    """

    def __init__(self, baseline_mask: np.ndarray, seed=42) -> None:
        """
        baseline_mask: original missing mask (convention: 1=observed, 0=missing)
        """
        self.rng = np.random.default_rng(seed)
        self.baseline_mask = baseline_mask.astype(int)
        self.current_mask = self.baseline_mask.copy()
        self.T, self.N = self.baseline_mask.shape
        self.total_elements = self.T * self.N

        # Track eval masks across rates
        self.cumulative_eval_mask = np.zeros_like(self.baseline_mask, dtype=bool)
        self.fixed_eval_mask: Optional[np.ndarray] = None

        # Track first rate info
        self.first_rate: Optional[float] = 0.0
        self.first_rate_eval_fraction: float = 0.0

        assert not np.shares_memory(self.baseline_mask, self.current_mask)

    def inject_block_to_rate(
        self,
        config: ScenarioConfig,
    ) -> ScenarioResult:
        count_observed_before = self.current_mask.sum()

        # Convert target missing rate to target observed rate
        target_missing_count = int(config.target_missing_rate * self.total_elements)
        target_observed_count = self.total_elements - target_missing_count

        if count_observed_before <= target_observed_count:
            # Already at or below target (enough missing)
            newly_injected = np.zeros_like(self.current_mask, dtype=bool)

            # Update cumulative eval mask (no change)
            self.cumulative_eval_mask = np.logical_or(
                self.cumulative_eval_mask, newly_injected
            )

            # Set fixed eval mask if first rate
            if self.fixed_eval_mask is None:
                self.fixed_eval_mask = newly_injected.copy()
                self.first_rate = config.target_missing_rate
                self.first_rate_eval_fraction = float(newly_injected.mean())

            return ScenarioResult(
                config=config,
                full_mask=self.current_mask.copy(),
                baseline_mask=self.baseline_mask.copy(),
                eval_mask_fixed=self.fixed_eval_mask.copy(),
                eval_mask_newly=newly_injected,
                eval_mask_cumulative=self.cumulative_eval_mask.copy(),
                metadata={
                    "target_rate": float(config.target_missing_rate),
                    "actual_rate": float(1.0 - self.current_mask.mean()),
                    "blocks_injected": [],
                    "status": "no_injection_needed",
                    "injection_mode": "cumulative",
                    "eval_mask_types": ["fixed", "newly", "cumulative"],
                    "eval_fraction": float(self.first_rate_eval_fraction or 0.0),
                },
            )

        self.current_mask, newly_injected, blocks_injected = _inject_block_core(
            self.current_mask.copy(),
            target_observed_count,
            config.block_size,
            self.rng,
        )

        # Update cumulative mask
        self.cumulative_eval_mask = np.logical_or(
            self.cumulative_eval_mask, newly_injected
        )
        # Set fixed eval mask at first rate
        if self.fixed_eval_mask is None:
            self.fixed_eval_mask = newly_injected.copy()
            self.first_rate = config.target_missing_rate
            self.first_rate_eval_fraction = float(newly_injected.mean())
            print(
                f"Fixed eval mask set at {config.target_missing_rate:.0%}: "
                f"{self.fixed_eval_mask.sum():,} positions ({self.first_rate_eval_fraction:.1%})"
            )

        actual_missing_rate = float(1.0 - self.current_mask.mean())

        return ScenarioResult(
            config=config,
            full_mask=self.current_mask.copy(),
            baseline_mask=self.baseline_mask.copy(),
            eval_mask_fixed=self.fixed_eval_mask.copy(),
            eval_mask_newly=newly_injected,
            eval_mask_cumulative=self.cumulative_eval_mask.copy(),
            metadata={
                "blocks_count": len(blocks_injected),
                "newly_injected_count": int(newly_injected.sum()),
                "actual_rate": float(actual_missing_rate),
                "blocks_injected": blocks_injected,
                "injection_mode": "cumulative",
                "eval_mask_types": ["fixed", "newly", "cumulative"],
                "eval_fraction": float(self.first_rate_eval_fraction),
                "eval_fixed_points": int(self.fixed_eval_mask.sum()),
                "eval_newly_points": int(newly_injected.sum()),
                "eval_cumulative_points": int(self.cumulative_eval_mask.sum()),
            },
        )


def _inject_block_core(
    current_mask: np.ndarray,
    target_observed_count: int,
    block_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """
    MCAR blocks with flexible sizes (1 to block_size).

    Every valid position has EQUAL probability, regardless of available size.
    Actual injected block sizes vary naturally (realistic fragmentation).
    """
    mask_before = current_mask.copy()
    T, N = current_mask.shape
    blocks_injected = []

    valid_map = current_mask.copy()
    kernel = np.ones(block_size)
    scores = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="valid"), axis=0, arr=valid_map
    )

    # Collect ALL positions (sizes 1 to block_size)
    all_blocks = []
    for sensor in range(N):
        for t_start in range(len(scores[:, sensor])):
            size = scores[t_start, sensor]
            if size > 0:
                all_blocks.append((int(t_start), int(sensor), int(size)))

    # FULLY SHUFFLE (equal probability for ALL positions)
    rng.shuffle(all_blocks)

    # Inject in random order
    for t_start, sensor, size in all_blocks:
        if current_mask.sum() <= target_observed_count:
            break

        t_end = min(t_start + size, T)
        if current_mask[t_start:t_end, sensor].sum() == size:
            current_mask[t_start:t_end, sensor] = 0
            blocks_injected.append([int(t_start), int(t_end), int(sensor)])

    newly_injected = ((mask_before == 1) & (current_mask == 0)).astype(int)
    return current_mask, newly_injected, blocks_injected


def _subsample_eval_mask(
    newly_injected: np.ndarray,
    target_eval_fraction: float,
    total_positions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Subsample newly injected positions to fixed eval mask size.

    For independent mode: ensures consistent eval size across rates.

    Parameters
    ----------
    newly_injected : np.ndarray
        Boolean mask of newly injected positions
    target_eval_fraction : float
        Desired fraction of total data for eval mask (e.g., 0.05)
    total_positions : int
        Total data positions (T × N)
    rng : np.random.Generator
        Random generator for reproducible subsampling

    Returns
    -------
    subsampled : np.ndarray
        Boolean mask with ~target_eval_fraction of total positions
    """
    target_eval_count = int(target_eval_fraction * total_positions)
    available_positions = np.where(newly_injected.flatten())[0]

    if len(available_positions) <= target_eval_count:
        # Not enough positions — return all available
        return newly_injected.copy()

    # Randomly subsample to target count
    selected_indices = rng.choice(
        available_positions, size=target_eval_count, replace=False
    )
    subsampled = np.zeros_like(newly_injected.flatten())
    subsampled[selected_indices] = 1

    return subsampled.reshape(newly_injected.shape)


def inject_mcar_blocks(
    config: ScenarioConfig,
    baseline_mask: np.ndarray,
    eval_fraction: Optional[float] = None,
) -> ScenarioResult:
    """
    Inject MCAR blocks (stateless, independent scenario)
    """
    rng = np.random.default_rng(config.seed)
    T, N = baseline_mask.shape
    total = T * N
    target_rate = config.target_missing_rate

    current_mask = baseline_mask.copy()
    target_observed_count = total - int(target_rate * total)

    if current_mask.sum() <= target_observed_count:
        newly_injected = np.zeros_like(current_mask, dtype=bool)
        actual_rate = float(1.0 - current_mask.mean())

        if eval_fraction is not None:
            eval_mask_fixed = _subsample_eval_mask(
                newly_injected=newly_injected,
                target_eval_fraction=eval_fraction,
                total_positions=total,
                rng=rng,
            )
            eval_mask_types = ["fixed_subsampled", "newly_full"]
        else:
            eval_mask_fixed = newly_injected
            eval_mask_types = ["rate_specific"]

        return ScenarioResult(
            config=config,
            full_mask=current_mask.copy(),
            baseline_mask=baseline_mask.copy(),
            eval_mask_fixed=eval_mask_fixed,
            eval_mask_newly=newly_injected,
            eval_mask_cumulative=newly_injected,
            metadata={
                "target_rate": float(target_rate),
                "actual_rate": float(1.0 - current_mask.mean()),
                "blocks_injected": [],
                "status": "no_injection_needed",
                "injection_mode": "independent",
                "eval_mask_types": eval_mask_types,
                "eval_fraction": float(eval_fraction or newly_injected.mean()),
            },
        )

    current_mask, newly_injected, blocks_injected = _inject_block_core(
        current_mask,
        target_observed_count,
        config.block_size,
        rng,
    )

    actual_rate = float(1.0 - current_mask.mean())

    # Determine eval masks (subsampled or not)
    if eval_fraction is not None:
        # Subsample to fixed eval fraction
        subsample_rng = np.random.default_rng(
            config.seed + hash(config.target_missing_rate) % 10000
        )
        eval_mask_fixed = _subsample_eval_mask(
            newly_injected=newly_injected,
            target_eval_fraction=eval_fraction,
            total_positions=total,
            rng=subsample_rng,
        )
        eval_mask_types = ["fixed_subsampled", "newly_full"]
    else:
        # Use newly injected as-is
        eval_mask_fixed = newly_injected
        eval_mask_types = ["rate_specific"]

    # For independent mode: all three eval masks are the same
    # (cumulative and fixed don't apply to single independent scenario)
    return ScenarioResult(
        config=config,
        full_mask=current_mask.copy(),
        baseline_mask=baseline_mask.copy(),
        eval_mask_fixed=eval_mask_fixed,
        eval_mask_newly=newly_injected,
        eval_mask_cumulative=newly_injected,
        metadata={
            "blocks_count": len(blocks_injected),
            "newly_injected_count": int(newly_injected.sum()),
            "actual_rate": float(actual_rate),
            "blocks_injected": blocks_injected,
            "injection_mode": "independent",
            "eval_mask_types": eval_mask_types,
            "eval_fraction": float(eval_fraction or newly_injected.mean()),
        },
    )


def inject_mcar_points(
    config: ScenarioConfig,
    baseline_mask: np.ndarray,
    eval_fraction: Optional[float] = None,
) -> ScenarioResult:
    """
    Inject MCAR point-wise missingness (block_size=1)
    """
    return inject_mcar_blocks(
        config=config,
        baseline_mask=baseline_mask,
        eval_fraction=eval_fraction,
    )
