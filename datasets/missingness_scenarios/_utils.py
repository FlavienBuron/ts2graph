import numpy as np


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
    selected_indices = rng.choice(available_positions, size=target_eval_count, replace=False)
    subsampled = np.zeros_like(newly_injected.flatten())
    subsampled[selected_indices] = 1

    return subsampled.reshape(newly_injected.shape)
