"""Random acquisition from the highest-scoring unacquired geometries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AcquisitionSelection:
    """Eligible candidates and newly acquired geometries for one round."""

    eligible_indices: np.ndarray
    acquired_indices: np.ndarray


def select_acquisitions(
    scores: np.ndarray,
    *,
    already_acquired: Iterable[int],
    acquired_per_round: int,
    acquire_from_uncertain_fraction: float,
    rng: np.random.Generator,
) -> AcquisitionSelection:
    """Randomly sample without replacement from the top unacquired fraction.

    Round the candidate count up and break score ties by global index.
    Acquire all candidates when fewer than the requested count are available.
    A zero fraction or fully acquired pool produces an empty selection.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("'scores' must be a non-empty one-dimensional array")
    if not np.isfinite(scores).all():
        raise ValueError("'scores' must contain only finite values")
    if (
        isinstance(acquired_per_round, bool)
        or not isinstance(acquired_per_round, (int, np.integer))
        or acquired_per_round < 1
    ):
        raise ValueError("'acquired_per_round' must be a positive integer")
    fraction = acquire_from_uncertain_fraction
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float, np.integer, np.floating))
        or not np.isfinite(fraction)
        or not 0.0 <= fraction <= 1.0
    ):
        raise ValueError("'acquire_from_uncertain_fraction' must be a finite number in [0, 1]")

    acquired = _indices(already_acquired, len(scores))
    available = np.ones(len(scores), dtype=bool)
    available[acquired] = False
    remaining = np.flatnonzero(available)
    ranked = remaining[np.argsort(-scores[remaining], kind="stable")]
    candidate_count = int(np.ceil(float(fraction) * len(remaining)))
    eligible = ranked[:candidate_count]
    count = min(int(acquired_per_round), candidate_count)
    selected = (
        np.sort(rng.choice(eligible, size=count, replace=False))
        if count else np.asarray([], dtype=int)
    )
    return AcquisitionSelection(eligible_indices=eligible, acquired_indices=selected)


def _indices(indices: Iterable[int], size: int) -> np.ndarray:
    resolved = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("Acquired indices must be integers")
        index = int(index)
        if not 0 <= index < size:
            raise ValueError(f"Acquired index {index} is outside [0, {size - 1}]")
        resolved.append(index)
    if len(set(resolved)) != len(resolved):
        raise ValueError("Acquired indices must be unique")
    return np.asarray(resolved, dtype=int)
