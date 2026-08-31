"""Spatially diverse, uncertainty-thresholded active-learning selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_DIHEDRAL_PERIOD_DEGREES = 180.0


@dataclass(frozen=True)
class AcquisitionSelection:
    """The deterministic selection outcome for one active-learning round."""

    eligible_indices: np.ndarray
    seed_indices: np.ndarray
    neighbour_indices: np.ndarray
    acquired_indices: np.ndarray


def select_acquisitions(
    scores: np.ndarray,
    coordinates: np.ndarray,
    *,
    already_acquired: Iterable[int],
    grid_shape: tuple[int, int],
    uncertainty_threshold: float,
    max_seeds: int,
) -> AcquisitionSelection:
    """Choose high-uncertainty diverse seeds and their grid neighbours.

    Only scores strictly above ``uncertainty_threshold`` are eligible. The
    first seed is the highest-scoring eligible point; later seeds maximize
    normalized distance from acquired and already chosen points, then break
    ties by uncertainty score and finally global index.
    """
    scores, coordinates = _validate_inputs(scores, coordinates, grid_shape)
    threshold = _non_negative_finite(uncertainty_threshold, "uncertainty_threshold")
    if isinstance(max_seeds, bool) or not isinstance(max_seeds, (int, np.integer)):
        raise ValueError("'max_seeds' must be a positive integer")
    if max_seeds < 1:
        raise ValueError("'max_seeds' must be a positive integer")

    acquired = _indices(already_acquired, len(scores))
    acquired_set = set(acquired.tolist())
    eligible = np.asarray(
        [
            index
            for index, score in enumerate(scores)
            if index not in acquired_set and score > threshold
        ],
        dtype=int,
    )
    seeds = _farthest_first(
        eligible,
        scores,
        coordinates,
        acquired,
        max_seeds=int(max_seeds),
    )
    seed_set = set(seeds.tolist())
    neighbours = _neighbours(seeds, grid_shape)
    neighbour_indices = np.asarray(
        sorted(set(neighbours.tolist()).difference(acquired_set, seed_set)),
        dtype=int,
    )
    acquired_indices = np.asarray(
        sorted(seed_set.union(neighbour_indices.tolist())), dtype=int
    )
    return AcquisitionSelection(
        eligible_indices=eligible,
        seed_indices=seeds,
        neighbour_indices=neighbour_indices,
        acquired_indices=acquired_indices,
    )


def _farthest_first(
    eligible: np.ndarray,
    scores: np.ndarray,
    coordinates: np.ndarray,
    already_acquired: np.ndarray,
    *,
    max_seeds: int,
) -> np.ndarray:
    if not len(eligible):
        return np.asarray([], dtype=int)
    selected = [max(eligible, key=lambda index: (scores[index], -index))]
    scale = _coordinate_scale(coordinates)
    while len(selected) < min(max_seeds, len(eligible)):
        references = np.asarray(
            [*already_acquired.tolist(), *selected], dtype=int
        )
        remaining = [index for index in eligible if index not in selected]
        selected.append(
            max(
                remaining,
                key=lambda index: (
                    _minimum_distance(
                        coordinates[index], coordinates[references], scale
                    ),
                    scores[index],
                    -index,
                ),
            )
        )
    return np.asarray(selected, dtype=int)


def _coordinate_scale(coordinates: np.ndarray) -> np.ndarray:
    ranges = np.ptp(coordinates, axis=0)
    return np.where(ranges > 0.0, ranges, 1.0)


def _minimum_distance(
    point: np.ndarray, references: np.ndarray, scale: np.ndarray
) -> float:
    deltas = np.abs(references - point)
    deltas[:, 1] = np.minimum(
        deltas[:, 1], _DIHEDRAL_PERIOD_DEGREES - deltas[:, 1]
    )
    normalized = deltas / scale
    return float(np.min(np.linalg.norm(normalized, axis=1)))


def _neighbours(seeds: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    cc_size, dihedral_size = grid_shape
    neighbours = set()
    for seed in seeds:
        cc_index, dihedral_index = divmod(int(seed), dihedral_size)
        for cc_neighbour in (cc_index - 1, cc_index + 1):
            if 0 <= cc_neighbour < cc_size:
                neighbours.add(cc_neighbour * dihedral_size + dihedral_index)
        for dihedral_neighbour in (
            (dihedral_index - 1) % dihedral_size,
            (dihedral_index + 1) % dihedral_size,
        ):
            neighbours.add(cc_index * dihedral_size + dihedral_neighbour)
    return np.asarray(sorted(neighbours), dtype=int)


def _validate_inputs(
    scores: np.ndarray, coordinates: np.ndarray, grid_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("'scores' must be a non-empty one-dimensional array")
    if coordinates.shape != (len(scores), 2):
        raise ValueError("'coordinates' must have shape (len(scores), 2)")
    if not np.isfinite(scores).all() or not np.isfinite(coordinates).all():
        raise ValueError("'scores' and 'coordinates' must contain only finite values")
    if (
        not isinstance(grid_shape, tuple)
        or len(grid_shape) != 2
        or any(
            isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1
            for size in grid_shape
        )
    ):
        raise ValueError("'grid_shape' must be a pair of positive integers")
    if int(np.prod(grid_shape)) != len(scores):
        raise ValueError("'grid_shape' must contain exactly one cell per score")
    return scores, coordinates


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


def _non_negative_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a non-negative finite number")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"'{name}' must be a non-negative finite number")
    return value
