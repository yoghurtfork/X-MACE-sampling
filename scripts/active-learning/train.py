"""K-fold committee training for resumable active-learning rounds."""

from __future__ import annotations

import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from scripts import data, evaluation, model, state


def train_k_fold_models(
    *,
    initial_model: torch.nn.Module,
    all_atoms: list[Any],
    test_atoms: list[Any],
    additional_test_sets: dict[str, list[Any]] | None = None,
    model_prefix: str,
    run_dir: Path,
    data_builder_class: Any,
    trainer_class: Any,
    tester: Any,
    loss_fn: torch.nn.Module,
    device: torch.device,
    seed: int,
    k: int,
    r_max: float,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    trainer_options: dict[str, Any],
    energy_key: str,
    forces_key: str,
    e0s: dict[str, float] | None,
    checkpoint_epochs: int | None = None,
    strategy: str = "naive",
    strategy_kwargs: dict[str, Any] | None = None,
    generate_plots: bool = True,
    progress_label: str | None = None,
    on_fold_complete: Callable[[dict[str, Any]], None] | None = None,
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and persist one deterministic active-learning committee."""
    if not 2 <= k <= len(all_atoms):
        raise ValueError(
            f"'k' must be between 2 and the {model_prefix} dataset size ({len(all_atoms)})"
        )
    if progress_label is not None and (
        not isinstance(progress_label, str) or not progress_label
    ):
        raise ValueError("'progress_label' must be a non-empty string or None")

    builder = data_builder_class(
        cutoff=r_max, energy_key=energy_key, forces_key=forces_key, E0s=e0s
    )
    builder.load(all_atoms, batch_size=batch_size, shuffle=False)
    resolved_e0s = data.resolved_e0s(builder)
    test_loaders = {
        "test_1": data.make_loader(builder, test_atoms, batch_size, shuffle=False)
    }
    test_loaders.update({
        name: data.make_loader(builder, atoms, batch_size, shuffle=False)
        for name, atoms in (additional_test_sets or {}).items()
    })

    _seed_everything(seed)
    started_at = time.time()
    fold_results: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, str]] = {}
    model_paths: dict[str, str] = {}
    for fold_number, (train_indices, valid_indices) in enumerate(
        data.kfold_splits(len(all_atoms), k, seed), start=1
    ):
        fold_seed = seed + fold_number
        _seed_everything(fold_seed)
        if progress_label is not None:
            print(
                f"{progress_label} | starting fold {fold_number}/{k} "
                f"(train={len(train_indices)}, validation={len(valid_indices)}, "
                f"seed={fold_seed})",
                flush=True,
            )
        train_loader = data.make_loader(
            builder, _atoms_at(all_atoms, train_indices), batch_size, shuffle=True
        )
        valid_loader = data.make_loader(
            builder, _atoms_at(all_atoms, valid_indices), batch_size, shuffle=False
        )
        model_key = f"model_{fold_number}"
        model_path = (Path(run_dir) / f"{model_prefix}_fold_{fold_number}.pt").resolve()
        fold_model = deepcopy(initial_model).to(device)
        trainer = trainer_class(
            max_epochs=max_epochs,
            device=device,
            **{**trainer_options, "optimiser_lr": learning_rate},
        )
        try:
            effective_strategy_kwargs = (
                strategy_kwargs
                if strategy_kwargs is not None
                else (
                    {"frozen_layers": ["node_embedding", "interactions"]}
                    if strategy == "freeze" else {}
                )
            )
            fold_model = model.apply_training_strategy(
                fold_model,
                {"strategy": strategy, "strategy_kwargs": effective_strategy_kwargs},
            ).to(device)
            fold_model, history = trainer.train_model(
                fold_model,
                train_loader,
                valid_loader,
                loss_fn,
                checkpoint_epoch=checkpoint_epochs,
            )
        except KeyboardInterrupt:
            saved_path = state.save_model(fold_model, model_path)
            if on_checkpoint is not None:
                on_checkpoint(_snapshot(
                    fold_results, artifacts,
                    {**model_paths, model_key: str(saved_path)},
                    total_folds=k, started_at=started_at,
                    include_artifacts=generate_plots,
                    current_fold={
                        "key": model_key, "fold_seed": fold_seed,
                        "status": "interrupted", "model_path": str(saved_path),
                    },
                ))
            raise

        saved_path = state.save_model(fold_model, model_path)
        checkpoint_metrics = _save_and_evaluate_checkpoints(
            fold_model, history, checkpoint_epochs, test_loaders, tester, device, saved_path
        )
        safe_history = dict(history)
        safe_history["checkpoint_models"] = checkpoint_metrics
        if on_checkpoint is not None:
            on_checkpoint(_snapshot(
                fold_results, artifacts,
                {**model_paths, model_key: str(saved_path)},
                total_folds=k, started_at=started_at,
                include_artifacts=generate_plots,
                current_fold={
                    "key": model_key, "fold_seed": fold_seed,
                    "status": "trained_pending_evaluation", "model_path": str(saved_path),
                    "history": safe_history,
                },
            ))
        test_metrics = evaluation.evaluate_test_sets(fold_model, test_loaders, tester)
        best_epoch = int(safe_history["best_epoch"])
        for metrics in test_metrics.values():
            metrics["best_epoch"] = best_epoch
        fold_results[model_key] = {
            "fold_seed": fold_seed, "history": safe_history,
            "metrics": test_metrics["test_1"], "test_metrics": test_metrics,
            "model_path": str(saved_path),
        }
        model_paths[model_key] = str(saved_path)
        if on_checkpoint is not None:
            on_checkpoint(_snapshot(
                fold_results, artifacts, model_paths, total_folds=k,
                started_at=started_at, include_artifacts=generate_plots,
                current_fold={
                    "key": model_key, "fold_seed": fold_seed,
                    "status": "evaluated_pending_artifacts", "model_path": str(saved_path),
                },
            ))
        if generate_plots:
            artifacts[model_key] = {
                "loss_plot": Path(state.save_loss_plot(
                    run_dir, safe_history,
                    title=f"{model_prefix.replace('_', ' ').title()} fold {fold_number}",
                    filename=f"{model_prefix}_fold_{fold_number}_loss.png",
                )).name,
                "validation_mae_plot": Path(state.save_epoch_mae_plot(
                    run_dir, safe_history,
                    title=(f"{model_prefix.replace('_', ' ').title()} fold {fold_number} "
                           "validation MAE"),
                    filename=f"{model_prefix}_fold_{fold_number}_validation_mae.png",
                )).name,
            }
        snapshot = _snapshot(
            fold_results, artifacts, model_paths, total_folds=k,
            started_at=started_at, include_artifacts=generate_plots,
        )
        if on_fold_complete is not None:
            on_fold_complete(snapshot)

    return _snapshot(
        fold_results, artifacts, model_paths, total_folds=k,
        started_at=started_at, include_artifacts=generate_plots,
    ) | {"E0s": resolved_e0s}


def _save_and_evaluate_checkpoints(
    trained_model: Any,
    history: dict[str, Any],
    checkpoint_epochs: int | None,
    test_loaders: dict[str, Any],
    tester: Any,
    device: torch.device,
    model_path: Path,
) -> list[dict[str, Any]]:
    checkpoints = evaluation.evaluate_checkpoint_models(
        trained_model, history, checkpoint_epochs, test_loaders, tester, device
    )
    results: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_model = checkpoint.pop("model")
        checkpoint_path = model_path.with_name(
            f"{model_path.stem}_checkpoint_epoch_{checkpoint['epoch']}{model_path.suffix}"
        )
        checkpoint["model_path"] = str(state.save_model(checkpoint_model, checkpoint_path))
        results.append(checkpoint)
    return results


def _snapshot(
    fold_results: dict[str, dict[str, Any]],
    artifacts: dict[str, dict[str, str]],
    model_paths: dict[str, str],
    *,
    total_folds: int,
    started_at: float,
    current_fold: dict[str, Any] | None = None,
    include_artifacts: bool,
) -> dict[str, Any]:
    if fold_results:
        best_epochs = np.asarray(
            [fold["metrics"]["best_epoch"] for fold in fold_results.values()], dtype=float
        )
        combined_validation: dict[str, Any] | None = {
            "best_epoch": [float(np.mean(best_epochs)), float(np.var(best_epochs))]
        }
        aggregate_by_set = {
            test_name: evaluation.aggregate_fold_metrics({
                fold_name: {"metrics": fold["test_metrics"][test_name]}
                for fold_name, fold in fold_results.items()
            })
            for test_name in next(iter(fold_results.values()))["test_metrics"]
        }
        aggregate_metrics: dict[str, Any] | None = aggregate_by_set["test_1"]
    else:
        combined_validation = None
        aggregate_metrics = None
        aggregate_by_set = None
    snapshot: dict[str, Any] = {
        "folds": fold_results,
        "combined_validation": combined_validation,
        "aggregate_test_metrics": aggregate_metrics,
        "aggregate_test_metrics_by_set": aggregate_by_set,
        "training_seconds": time.time() - started_at,
        "model_paths": model_paths,
        "completed_folds": len(fold_results),
        "total_folds": total_folds,
    }
    if include_artifacts:
        snapshot["artifacts"] = artifacts
    if current_fold is not None:
        snapshot["current_fold"] = current_fold
    return snapshot


def _atoms_at(atoms: list[Any], indices: np.ndarray) -> list[Any]:
    return [atoms[int(index)] for index in indices]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
