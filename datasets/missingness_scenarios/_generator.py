from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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
            return ScenarioResult(
                config=config,
                full_mask=self.current_mask.copy(),
                eval_mask=np.zeros_like(self.current_mask).astype(bool),
                baseline_mask=self.baseline_mask.copy(),
                metadata={
                    "target_rate": float(config.target_missing_rate),
                    "actual_rate": float(1.0 - self.current_mask.mean()),
                    "blocks_injected": [],
                    "status": "no_injection_needed",
                },
            )

        self.current_mask, newly_injected, blocks_injected = _inject_block_core(
            self.current_mask.copy(),
            target_observed_count,
            config.block_size,
            self.rng,
        )

        actual_missing_rate = float(1.0 - self.current_mask.mean())

        return ScenarioResult(
            config=config,
            full_mask=self.current_mask.copy(),
            eval_mask=newly_injected,
            baseline_mask=self.baseline_mask,
            metadata={
                "blocks_count": len(blocks_injected),
                "newly_injected_count": int(newly_injected.sum()),
                "actual_rate": float(actual_missing_rate),
                "blocks_injected": blocks_injected,
                "cumulative": True,
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

    # ✅ FULLY SHUFFLE (equal probability for ALL positions)
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


# def _inject_block_core(
#     current_mask: np.ndarray,
#     target_observed_count: int,
#     block_size: int,
#     rng: np.random.Generator,
#     max_iterations: int = 100000,
# ) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
#     mask_before = current_mask.copy()
#     T = current_mask.shape[0]
#
#     blocks_injected = []
#     iterations = 0
#
#     # Inject until reaching target missing rate
#     while current_mask.sum() > target_observed_count and iterations < max_iterations:
#         iterations += 1
#
#         # Valid map: 1 = present, can be made missing, 0 = already missing
#         valid_map = current_mask.copy()
#
#         kernel = np.ones(block_size)
#         scores = np.apply_along_axis(
#             lambda x: np.convolve(x, kernel, mode="valid"), axis=0, arr=valid_map
#         )
#
#         # Priority 1: Full blocks can be injected (all positions can be made missing)
#         full_coords = np.argwhere(scores == block_size)
#         if len(full_coords) > 0:
#             shuffle_idxs = rng.permutation(len(full_coords))
#             full_coords = full_coords[shuffle_idxs]
#
#             for coord in full_coords[: min(100, len(full_coords))]:
#                 if current_mask.sum() <= target_observed_count:
#                     break
#                 t_start, series_id = coord
#                 t_end = min(t_start + block_size, T)
#
#                 if current_mask[t_start:t_end, series_id].sum() == block_size:
#                     current_mask[t_start:t_end, series_id] = 0
#                     blocks_injected.append([int(t_start), int(t_end), int(series_id)])
#             continue
#         # Priority 2: best partial fit
#         max_score = scores.max() if scores.size > 0 else 0
#         if max_score == 0:
#             break
#
#         partial_coords = np.argwhere(scores == max_score)
#         shuffle_idx = rng.permutation(len(partial_coords))
#         partial_coords = partial_coords[shuffle_idx]
#
#         for coord in partial_coords[: min(100, len(partial_coords))]:
#             if current_mask.sum() <= target_observed_count:
#                 break
#
#             t_start, series_id = coord
#             t_end = min(t_start + max_score, T)
#
#             if current_mask[t_start:t_end, series_id].sum() == max_score:
#                 current_mask[t_start:t_end, series_id] = 0
#                 blocks_injected.append([int(t_start), int(t_end), int(series_id)])
#
#     newly_injected = (mask_before == 1) & (current_mask == 0).astype(int)
#
#     return current_mask, newly_injected, blocks_injected


def inject_mcar_blocks(
    config: ScenarioConfig,
    baseline_mask: np.ndarray,
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
        return ScenarioResult(
            config=config,
            full_mask=current_mask.copy(),
            eval_mask=np.zeros_like(current_mask),
            baseline_mask=baseline_mask.copy(),
            metadata={
                "target_rate": target_rate,
                "actual_rate": float(1.0 - current_mask.mean()),
                "blocks_injected": [],
                "status": "no_injection_needed",
            },
        )

    current_mask, eval_mask, blocks_injected = _inject_block_core(
        current_mask,
        target_observed_count,
        config.block_size,
        rng,
    )

    return ScenarioResult(
        config=config,
        full_mask=current_mask.copy(),
        eval_mask=eval_mask,
        baseline_mask=baseline_mask,
        metadata={
            "blocks_count": len(blocks_injected),
            "newly_injected_count": int(eval_mask.sum()),
            "actual_rate": float(1.0 - current_mask.mean()),
            "blocks_injected": blocks_injected,
            "cumulative": True,
        },
    )


def inject_mcar_points(
    config: ScenarioConfig,
    baseline_mask: np.ndarray,
) -> ScenarioResult:
    """
    Inject MCAR point-wise missingness (block_size=1)
    """
    return inject_mcar_blocks(config=config, baseline_mask=baseline_mask)
