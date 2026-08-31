"""Committee uncertainty calculations for active acquisition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommitteeUncertainty:
    """Per-configuration committee disagreement summaries."""

    energy_std: np.ndarray
    force_std: np.ndarray
    score: np.ndarray


def committee_uncertainty(
    energies: np.ndarray,
    forces: np.ndarray,
    *,
    energy_weight: float,
    force_weight: float,
) -> CommitteeUncertainty:
    """Compute weighted energy-and-force standard-deviation uncertainty.

    Arrays must have leading dimensions ``(n_models, n_configurations)``.
    Any remaining dimensions are prediction components: electronic states for
    energies, and states/atoms/Cartesian components for forces. Their
    committee standard deviations are averaged into one value per geometry.
    """
    energies = _prediction_array(energies, "energies")
    forces = _prediction_array(forces, "forces")
    if energies.shape[:2] != forces.shape[:2]:
        raise ValueError(
            "Energy and force predictions must agree on model and configuration counts"
        )
    energy_weight = _weight(energy_weight, "energy_weight")
    force_weight = _weight(force_weight, "force_weight")
    if energy_weight == 0.0 and force_weight == 0.0:
        raise ValueError("At least one uncertainty weight must be positive")

    energy_std = _mean_prediction_std(energies)
    force_std = _mean_prediction_std(forces)
    score = energy_weight * energy_std + force_weight * force_std
    return CommitteeUncertainty(
        energy_std=energy_std,
        force_std=force_std,
        score=score,
    )


def _prediction_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim < 2:
        raise ValueError(
            f"'{name}' must have shape (n_models, n_configurations, ...)"
        )
    if array.shape[0] < 2:
        raise ValueError("Committee uncertainty requires at least two models")
    if array.shape[1] < 1:
        raise ValueError("Committee predictions must include at least one configuration")
    if not np.isfinite(array).all():
        raise ValueError(f"'{name}' must contain only finite values")
    return array


def _mean_prediction_std(values: np.ndarray) -> np.ndarray:
    """Average population committee standard deviations over output components."""
    std = np.std(values, axis=0)
    component_axes = tuple(range(1, std.ndim))
    return np.mean(std, axis=component_axes) if component_axes else std


def _weight(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a non-negative number")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"'{name}' must be a non-negative finite number")
    return value
