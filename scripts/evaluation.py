"""Read-only training and checkpoint evaluation helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def evaluate_model(model: Any, loader: Any, tester: Any) -> dict[str, Any]:
    """Evaluate one model and return scalar plus per-electronic-state MAEs."""
    model.eval()
    tester.run_test(model, loader)
    return {
        "energy_mae_ev": float(tester.get_energy_mae()),
        "force_mae_ev_per_ang": float(tester.get_force_mae()),
        "energy_mae_by_state_ev": maes_by_state(
            tester.get_energy_mae_by_state(), "energy"
        ),
        "force_mae_by_state_ev_per_ang": maes_by_state(
            tester.get_force_mae_by_state(), "force"
        ),
    }


def evaluate_test_sets(
    model: Any, test_loaders: dict[str, Any], tester: Any
) -> dict[str, dict[str, Any]]:
    """Evaluate a model against every named test loader."""
    return {
        name: evaluate_model(model, loader, tester)
        for name, loader in test_loaders.items()
    }


def evaluate_checkpoint_models(
    model: Any,
    history: dict[str, Any],
    checkpoint_epochs: int | None,
    test_loaders: dict[str, Any],
    tester: Any,
    device: Any,
) -> list[dict[str, Any]]:
    """Evaluate trainer checkpoint states without writing models or files.

    Returned entries retain their in-memory model under ``model`` so the
    orchestration layer can persist it through ``state.py`` and then remove it
    before placing the remaining JSON-safe information in the result.
    """
    checkpoint_states = history.get("checkpoint_models", [])
    if checkpoint_epochs is None:
        if checkpoint_states:
            raise ValueError("Trainer returned checkpoint models without checkpoint_epochs")
        return []
    results: list[dict[str, Any]] = []
    for number, checkpoint_state in enumerate(checkpoint_states, start=1):
        checkpoint_model = deepcopy(model).to(device)
        checkpoint_model.load_state_dict(checkpoint_state)
        metrics_by_set = evaluate_test_sets(checkpoint_model, test_loaders, tester)
        primary = metrics_by_set["test_1"]
        results.append(
            {
                "epoch": number * checkpoint_epochs,
                "model": checkpoint_model,
                "test_energy_mae": primary["energy_mae_ev"],
                "test_force_mae": primary["force_mae_ev_per_ang"],
                "test_metrics": metrics_by_set,
            }
        )
    return results


def aggregate_fold_metrics(fold_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Calculate mean/variance metrics over completed validation folds."""
    if not fold_results:
        raise ValueError("Cannot aggregate metrics for zero folds")
    metrics = [fold["metrics"] for fold in fold_results.values()]
    result: dict[str, Any] = {}
    for key in ("energy_mae_ev", "force_mae_ev_per_ang"):
        values = np.asarray([metric[key] for metric in metrics], dtype=float)
        result[key] = {"mean": float(np.mean(values)), "variance": float(np.var(values))}
    for key in ("energy_mae_by_state_ev", "force_mae_by_state_ev_per_ang"):
        states = list(metrics[0][key])
        if any(list(metric[key]) != states for metric in metrics[1:]):
            raise ValueError(f"Cross-validation folds returned inconsistent states for '{key}'")
        result[key] = {
            state: _mean_and_variance([metric[key][state] for metric in metrics])
            for state in states
        }
    return result


def maes_by_state(values: Any, metric_name: str) -> dict[str, float]:
    """Label tester-provided per-state MAEs as ``S0``, ``S1``, and so on."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    if isinstance(values, dict):
        values = list(values.values())
    flattened = np.asarray(values, dtype=float).reshape(-1)
    if not len(flattened):
        raise ValueError(f"Tester returned no {metric_name} MAEs")
    if not np.all(np.isfinite(flattened)):
        raise ValueError(f"Tester returned non-finite {metric_name} MAEs")
    return {f"S{index}": float(value) for index, value in enumerate(flattened)}


def _mean_and_variance(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "variance": float(np.var(array))}
