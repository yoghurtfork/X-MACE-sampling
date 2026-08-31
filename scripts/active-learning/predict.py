"""Label-free committee inference over unacquired HF-grid geometries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class CommitteePredictions:
    """Aligned energy and force predictions from every committee model."""

    energies: np.ndarray
    forces: np.ndarray

    @property
    def n_models(self) -> int:
        return int(self.energies.shape[0])

    @property
    def n_configurations(self) -> int:
        return int(self.energies.shape[1])


def predict_committee(
    models: Iterable[torch.nn.Module], loader: Any, *, device: torch.device
) -> CommitteePredictions:
    """Predict one target-free loader with every committee model.

    ``loader`` must preserve candidate order and must not be shuffled. This
    function accesses only model outputs and batch graph boundaries; it never
    reads batch reference energies or forces.
    """
    models = list(models)
    if len(models) < 2:
        raise ValueError("Committee prediction requires at least two models")

    predictions = [_predict_model(model, loader, device) for model in models]
    energies = _stack_predictions(
        [prediction[0] for prediction in predictions], "energy"
    )
    forces = _stack_predictions(
        [prediction[1] for prediction in predictions], "force"
    )
    if energies.shape[1] != forces.shape[1]:
        raise ValueError(
            "Committee energy and force predictions disagree on configuration count"
        )
    return CommitteePredictions(energies=energies, forces=forces)


def _predict_model(
    model: torch.nn.Module, loader: Any, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()
    energy_batches: list[np.ndarray] = []
    force_configurations: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), training=False, compute_force=True)
        pointers = _batch_pointers(batch)
        energy = output["energy"].detach().cpu().numpy()
        if energy.ndim == 0 or energy.shape[0] != len(pointers) - 1:
            raise ValueError(
                "Model energy output must have one leading entry per configuration"
            )
        forces = output["forces"].detach().cpu().numpy()
        if forces.ndim < 1 or forces.shape[0] != int(pointers[-1]):
            raise ValueError(
                "Model force output must have one leading entry per atom"
            )
        energy_batches.append(energy)
        force_configurations.extend(
            forces[start:stop] for start, stop in zip(pointers[:-1], pointers[1:])
        )

    if not energy_batches:
        raise ValueError("Prediction loader must contain at least one configuration")
    energies = np.concatenate(energy_batches, axis=0)
    if len(force_configurations) != len(energies):
        raise ValueError("Model energy and force outputs are not configuration-aligned")
    try:
        return energies, np.stack(force_configurations, axis=0)
    except ValueError as error:
        raise ValueError(
            "All prediction geometries must have matching force-output shapes"
        ) from error


def _batch_pointers(batch: Any) -> np.ndarray:
    if not hasattr(batch, "ptr"):
        raise ValueError("Prediction batch must expose PyG graph pointers through 'ptr'")
    pointers = batch.ptr.detach().cpu().numpy().astype(int, copy=False)
    if pointers.ndim != 1 or len(pointers) < 2 or pointers[0] != 0:
        raise ValueError("Prediction batch has invalid graph pointers")
    if np.any(np.diff(pointers) < 1):
        raise ValueError("Prediction batch contains an empty graph")
    return pointers


def _stack_predictions(predictions: list[np.ndarray], name: str) -> np.ndarray:
    try:
        return np.stack(predictions, axis=0)
    except ValueError as error:
        raise ValueError(
            f"Committee {name} predictions have inconsistent output shapes"
        ) from error
