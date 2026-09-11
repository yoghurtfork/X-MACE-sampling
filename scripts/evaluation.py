"""Read-only training and checkpoint evaluation helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch


def evaluate_model(
    model: Any, loader: Any, tester: Any, *, compute_nacs: bool = False
) -> dict[str, Any]:
    """Evaluate one model and return scalar plus per-electronic-state MAEs."""
    model.eval()
    tester.run_test(model, loader, compute_nacs=compute_nacs)
    metrics = {
        "energy_mae_ev": float(tester.get_energy_mae()),
        "force_mae_ev_per_ang": float(tester.get_force_mae()),
        "energy_mae_by_state_ev": maes_by_state(
            tester.get_energy_mae_by_state(), "energy"
        ),
        "force_mae_by_state_ev_per_ang": maes_by_state(
            tester.get_force_mae_by_state(), "force"
        ),
    }
    if compute_nacs:
        phase_nac_by_state = tester.get_nac_phase_rmse_by_pair()
        abs_nac_by_state = tester.get_nac_abs_mae_by_pair()
        metrics.update(
            {
                "nac_phase_rmse": float(tester.get_nac_phase_rmse()),
                "nac_abs_mae": float(tester.get_nac_abs_mae()),
                "nac_phase_rmse_by_pair": np.asarray(
                    phase_nac_by_state, dtype=float
                ).reshape(-1).tolist(),
                "nac_abs_mae_by_pair": np.asarray(
                    abs_nac_by_state, dtype=float
                ).reshape(-1).tolist(),
            }
        )
    return metrics


def evaluate_test_sets(
    model: Any, test_loaders: dict[str, Any], tester: Any, *, compute_nacs: bool = False
) -> dict[str, dict[str, Any]]:
    """Evaluate a model against every named test loader."""
    return {
        name: evaluate_model(model, loader, tester, compute_nacs=compute_nacs)
        for name, loader in test_loaders.items()
    }


def evaluate_checkpoint_models(
    model: Any,
    history: dict[str, Any],
    checkpoint_epochs: int | None,
    test_loaders: dict[str, Any],
    tester: Any,
    device: Any,
    *,
    compute_nacs: bool = False,
) -> list[dict[str, Any]]:
    """Load and evaluate every checkpoint recorded by the trainer."""
    checkpoint_entries = history.get("checkpoint_models", [])
    if checkpoint_epochs is None:
        if checkpoint_entries:
            raise ValueError("Trainer returned checkpoint models without checkpoint_epochs")
        return []
    results: list[dict[str, Any]] = []
    for checkpoint in checkpoint_entries:
        epoch = int(checkpoint["epoch"])
        checkpoint_path = str(checkpoint["path"])
        checkpoint_model = deepcopy(model).to(device)
        checkpoint_state = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        checkpoint_model.load_state_dict(checkpoint_state)
        metrics_by_set = evaluate_test_sets(
            checkpoint_model, test_loaders, tester, compute_nacs=compute_nacs
        )
        primary = metrics_by_set["test_1"]
        result = {
            "epoch": epoch,
            "model_path": checkpoint_path,
            "test_energy_mae": primary["energy_mae_ev"],
            "test_force_mae": primary["force_mae_ev_per_ang"],
            "test_metrics": metrics_by_set,
        }
        if compute_nacs:
            result.update(
                {
                    "test_nac_phase_rmse": primary["nac_phase_rmse"],
                    "test_nac_abs_mae": primary["nac_abs_mae"],
                }
            )
        results.append(result)
    return results


def aggregate_fold_metrics(
    fold_results: dict[str, dict[str, Any]], *, compute_nacs: bool = False
) -> dict[str, Any]:
    """Calculate mean/variance metrics over completed validation folds."""
    if not fold_results:
        raise ValueError("Cannot aggregate metrics for zero folds")
    metrics = [fold["metrics"] for fold in fold_results.values()]
    result: dict[str, Any] = {}
    scalar_keys = ["energy_mae_ev", "force_mae_ev_per_ang"]
    per_item_keys = ["energy_mae_by_state_ev", "force_mae_by_state_ev_per_ang"]
    if compute_nacs:
        scalar_keys.extend(["nac_phase_rmse", "nac_abs_mae"])
        per_item_keys.extend(["nac_phase_rmse_by_pair", "nac_abs_mae_by_pair"])
    for key in scalar_keys:
        values = np.asarray([metric[key] for metric in metrics], dtype=float)
        result[key] = {"mean": float(np.mean(values)), "variance": float(np.var(values))}
    for key in per_item_keys:
        first = metrics[0][key]
        if isinstance(first, dict):
            states = list(first)
            if any(list(metric[key]) != states for metric in metrics[1:]):
                raise ValueError(f"Cross-validation folds returned inconsistent states for '{key}'")
            result[key] = {
                state: _mean_and_variance([metric[key][state] for metric in metrics])
                for state in states
            }
        else:
            values = np.asarray(first, dtype=float).reshape(-1)
            if any(
                np.asarray(metric[key], dtype=float).reshape(-1).shape != values.shape
                for metric in metrics[1:]
            ):
                raise ValueError(f"Cross-validation folds returned inconsistent values for '{key}'")
            result[key] = [
                _mean_and_variance([
                    float(np.asarray(metric[key], dtype=float).reshape(-1)[index])
                    for metric in metrics
                ])
                for index in range(len(values))
            ]
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
