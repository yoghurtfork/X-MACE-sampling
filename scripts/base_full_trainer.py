"""Train base and full high-fidelity X-MACE models from JSON configurations.

This command reads scratch-training configurations (``transfer_learning:
false``) from the same input directory as ``tester.py`` and writes to the same
sequential ``run_<index>`` output directory.

Required keys are ``base_xyz``, ``transfer_xyz``, ``base_test_xyz``, and
``transfer_test_xyz``. Scratch epoch/learning-rate overrides and K-fold
cross-validation use the same JSON settings previously supported by
``tester.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tester import (
    BATCH_SIZE,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEVICE,
    MAX_EPOCHS,
    PATIENCE,
    R_MAX,
    SEED,
    VALIDATION_FRACTION,
    _evaluate,
    _import_project_modules,
    _next_run_dir,
    _path,
    _read_atoms,
    _required,
    _save_epoch_mae_plot,
    _save_loss_plot,
    _save_fold_selection_plot,
    _train_k_fold_models,
    _validate_device,
    _write_json,
)

SCRATCH_LR = 1.0e-3


def _save_scratch_mae_plot(
    run_dir: Path,
    base_metrics: dict[str, Any],
    full_metrics: dict[str, Any],
    *,
    cross_validation: bool,
) -> dict[str, str]:
    categories = ["Base model", "Full HF model"]
    colors = ["#377eb8", "#ff7f00"]
    plots = {}
    for metric, ylabel, filename, artifact_key in (
        (
            "energy_mae_ev",
            "Mean energy MAE, eV",
            "scratch_energy_mae_comparison.png",
            "energy_mae_comparison_plot",
        ),
        (
            "force_mae_ev_per_ang",
            "Mean force MAE, eV/Å",
            "scratch_force_mae_comparison.png",
            "force_mae_comparison_plot",
        ),
    ):
        if cross_validation:
            values = [
                base_metrics[metric]["mean"],
                full_metrics[metric]["mean"],
            ]
            errors = [
                np.sqrt(base_metrics[metric]["variance"]),
                np.sqrt(full_metrics[metric]["variance"]),
            ]
        else:
            values = [base_metrics[metric], full_metrics[metric]]
            errors = None
        fig, ax = plt.subplots()
        bars = ax.bar(
            categories,
            values,
            yerr=errors,
            capsize=5 if cross_validation else 0,
            color=colors,
        )
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set(ylabel=ylabel, title="Models trained from scratch")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=150)
        plt.close(fig)
        plots[artifact_key] = filename
    return plots


def _train_model(
    *,
    train_atoms: list[Any],
    valid_atoms: list[Any],
    test_atoms: list[Any],
    builder_class: Any,
    trainer_class: Any,
    initialise_autoencoder: Any,
    tester: Any,
    loss_fn: torch.nn.Module,
    device: torch.device,
    seed: int,
    r_max: float,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    early_stopping: bool,
    patience: int,
    restore_best: bool,
    verbose: bool,
    energy_key: str,
    forces_key: str,
    training: Any,
) -> dict[str, Any]:
    builder = builder_class(
        cutoff=r_max, energy_key=energy_key, forces_key=forces_key
    )
    train_loader = builder.load(
        train_atoms, batch_size=batch_size, shuffle=True
    )
    valid_loader = builder.load(
        valid_atoms, batch_size=batch_size, shuffle=False
    )
    test_loader = builder.load(
        test_atoms, batch_size=batch_size, shuffle=False
    )
    training.seed_everything(seed)
    model = initialise_autoencoder(
        builder.get_metadata(), preset="default_ani", load_base="ani500k"
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = trainer_class(
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        patience=patience,
        restore_best=restore_best,
        device=device,
        verbose=verbose,
    )
    started_at = time.time()
    model, history = trainer.train_model(
        model, train_loader, valid_loader, optimizer, loss_fn
    )
    metrics = _evaluate(model, test_loader, tester)
    metrics["best_epoch"] = int(history["best_epoch"])
    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "training_seconds": time.time() - started_at,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
    }


def run_config(config_path: Path, output_dir: Path) -> Path:
    (
        modules,
        AtomDataLoaderBuilder,
        Tester,
        _extract_latent_space,
        _NaiveStrategy,
        Trainer,
        initialise_autoencoder,
        descriptors,
        _selectors,
        training,
    ) = _import_project_modules()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The top-level JSON value must be an object")
    if config.get("transfer_learning", True) is not False:
        raise ValueError(
            "base_full_trainer.py requires 'transfer_learning': false"
        )

    run_dir = _next_run_dir(output_dir)
    result_path = run_dir / "result.json"
    result: dict[str, Any] = {
        "status": "running",
        "input_file": str(config_path.resolve()),
        "run_directory": str(run_dir.resolve()),
        "program": "base_full_trainer.py",
    }
    _write_json(result_path, result)

    try:
        seed = int(config.get("seed", SEED))
        cross_validation = config.get("cross_validation", False)
        if not isinstance(cross_validation, bool):
            raise ValueError("'cross_validation' must be a JSON boolean")
        if cross_validation:
            if (
                "k" not in config
                or isinstance(config["k"], bool)
                or not isinstance(config["k"], int)
            ):
                raise ValueError(
                    "'k' must be a JSON integer when cross-validation is true"
                )
            k = config["k"]
            if k < 2:
                raise ValueError("'k' must be at least 2")
        else:
            k = None

        max_epochs = int(config.get("max_epochs", MAX_EPOCHS))
        base_max_epochs = int(config.get("base_max_epochs", max_epochs))
        full_max_epochs = int(config.get("full_max_epochs", max_epochs))
        base_lr = float(config.get("base_learning_rate", SCRATCH_LR))
        full_lr = float(config.get("full_learning_rate", SCRATCH_LR))
        r_max = float(config.get("r_max", R_MAX))
        batch_size = int(config.get("batch_size", BATCH_SIZE))
        validation_fraction = float(
            config.get("validation_fraction", VALIDATION_FRACTION)
        )
        patience = int(config.get("patience", PATIENCE))
        device = _validate_device(str(config.get("device", DEVICE)))
        if min(max_epochs, base_max_epochs, full_max_epochs, batch_size) < 1:
            raise ValueError("Epoch counts and batch size must be positive")
        if min(base_lr, full_lr) <= 0:
            raise ValueError("Learning rates must be positive")
        if not cross_validation and not 0 < validation_fraction < 1:
            raise ValueError("'validation_fraction' must be between 0 and 1")

        base_atoms = _read_atoms(
            _path(_required(config, "base_xyz"), config_path),
            config.get("base_n_geometries"),
        )
        full_atoms = _read_atoms(
            _path(_required(config, "transfer_xyz"), config_path),
            config.get("transfer_n_geometries"),
        )
        base_test_atoms = _read_atoms(
            _path(_required(config, "base_test_xyz"), config_path)
        )
        full_test_atoms = _read_atoms(
            _path(_required(config, "transfer_test_xyz"), config_path)
        )
        if len(base_atoms) != len(full_atoms):
            raise ValueError("Base and transfer datasets must be aligned")

        warnings = []
        transfer_only = {
            "base_model_path",
            "full_model_path",
            "descriptor",
            "descriptor_kwargs",
            "selector",
            "selector_kwargs",
            "n_samples",
            "pca",
            "transfer_lr",
        }
        ignored = sorted(transfer_only.intersection(config))
        if ignored:
            warnings.append(
                "Ignoring transfer-learning-only configuration keys: "
                + ", ".join(ignored)
            )
        if cross_validation and "validation_fraction" in config:
            warnings.append(
                "Ignoring 'validation_fraction' because cross-validation "
                "replaces the fixed validation split"
            )
        if not cross_validation and "k" in config:
            warnings.append(
                "Ignoring 'k' because cross-validation is disabled"
            )
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)

        tester = Tester(device=device)
        loss_defaults = {
            "energy_weight": 1.0,
            "forces_weight": 5.0,
            "dipoles_weight": 0.0,
            "nacs_weight": 0.0,
            "socs_weight": 0.0,
        }
        loss_defaults.update(config.get("loss_kwargs", {}))
        loss_fn = modules.InvariantsWeightedEnergyForcesNacsDipoleLoss(
            **loss_defaults
        ).to(device)
        energy_key = str(config.get("energy_key", "REF_energy"))
        forces_key = str(config.get("forces_key", "REF_forces"))
        train_options = {
            "early_stopping": bool(config.get("early_stopping", True)),
            "patience": patience,
            "restore_best": bool(config.get("restore_best", True)),
            "verbose": bool(config.get("verbose", True)),
        }

        if cross_validation:
            common = {
                "run_dir": run_dir,
                "data_builder_class": AtomDataLoaderBuilder,
                "trainer_class": Trainer,
                "tester": tester,
                "loss_fn": loss_fn,
                "device": device,
                "seed": seed,
                "k": k,
                "r_max": r_max,
                "batch_size": batch_size,
                "energy_key": energy_key,
                "forces_key": forces_key,
                "training": training,
                **train_options,
            }
            initial_models = {}
            for name, atoms in (("base", base_atoms), ("full", full_atoms)):
                builder = AtomDataLoaderBuilder(
                    cutoff=r_max,
                    energy_key=energy_key,
                    forces_key=forces_key,
                )
                builder.load(atoms, batch_size=batch_size, shuffle=False)
                training.seed_everything(seed)
                initial_models[name] = initialise_autoencoder(
                    builder.get_metadata(), preset="default_ani", load_base="ani500k"
                ).to(device)
            base_run = _train_k_fold_models(
                initial_model=initial_models["base"],
                all_atoms=base_atoms,
                test_atoms=base_test_atoms,
                model_prefix="base_model",
                max_epochs=base_max_epochs,
                learning_rate=base_lr,
                **common,
            )
            full_run = _train_k_fold_models(
                initial_model=initial_models["full"],
                all_atoms=full_atoms,
                test_atoms=full_test_atoms,
                model_prefix="full_model",
                max_epochs=full_max_epochs,
                learning_rate=full_lr,
                **common,
            )
            for model_prefix, atoms, test_atoms, model_run in (
                ("base_model", base_atoms, base_test_atoms, base_run),
                ("full_model", full_atoms, full_test_atoms, full_run),
            ):
                splitter = KFold(
                    n_splits=k, shuffle=True, random_state=seed
                )
                all_indices = np.arange(len(atoms))
                for fold_number, (
                    fold_train_indices,
                    fold_valid_indices,
                ) in enumerate(
                    splitter.split(range(len(atoms))), start=1
                ):
                    model_key = f"model_{fold_number}"
                    model_run["artifacts"][model_key][
                        "selection_plot"
                    ] = _save_fold_selection_plot(
                        run_dir=run_dir,
                        base_atoms=atoms,
                        base_test_atoms=test_atoms,
                        sampled_global_indices=all_indices,
                        fold_train_indices=fold_train_indices,
                        fold_valid_indices=fold_valid_indices,
                        fold_number=fold_number,
                        model_prefix=model_prefix,
                        descriptors=descriptors,
                    )
            comparison_plots = _save_scratch_mae_plot(
                run_dir,
                base_run["aggregate_test_metrics"],
                full_run["aggregate_test_metrics"],
                cross_validation=True,
            )
            result.update(
                {
                    "status": "completed",
                    "config": config,
                    "transfer_learning": False,
                    "cross_validation": True,
                    "k": k,
                    "warnings": warnings,
                    "seed": seed,
                    "device": str(device),
                    "dataset_sizes": {
                        "base": len(base_atoms),
                        "transfer": len(full_atoms),
                        "base_test": len(base_test_atoms),
                        "transfer_test": len(full_test_atoms),
                    },
                    "metrics": {
                        "base_models": base_run["aggregate_test_metrics"],
                        "full_high_fidelity_models": full_run[
                            "aggregate_test_metrics"
                        ],
                    },
                    "cross_validation_training": {
                        "base_models": base_run,
                        "full_high_fidelity_models": full_run,
                    },
                    "models": {
                        "base_models": base_run["model_paths"],
                        "full_high_fidelity_models": full_run["model_paths"],
                    },
                    "artifacts": {
                        "base_models": base_run["artifacts"],
                        "full_high_fidelity_models": full_run["artifacts"],
                        **comparison_plots,
                    },
                }
            )
        else:
            indices = np.arange(len(base_atoms))
            train_indices, valid_indices = train_test_split(
                indices,
                test_size=validation_fraction,
                random_state=seed,
                shuffle=True,
            )
            common = {
                "builder_class": AtomDataLoaderBuilder,
                "trainer_class": Trainer,
                "initialise_autoencoder": initialise_autoencoder,
                "tester": tester,
                "loss_fn": loss_fn,
                "device": device,
                "seed": seed,
                "r_max": r_max,
                "batch_size": batch_size,
                "energy_key": energy_key,
                "forces_key": forces_key,
                "training": training,
                **train_options,
            }
            base_run = _train_model(
                train_atoms=[base_atoms[i] for i in train_indices],
                valid_atoms=[base_atoms[i] for i in valid_indices],
                test_atoms=base_test_atoms,
                max_epochs=base_max_epochs,
                learning_rate=base_lr,
                **common,
            )
            full_run = _train_model(
                train_atoms=[full_atoms[i] for i in train_indices],
                valid_atoms=[full_atoms[i] for i in valid_indices],
                test_atoms=full_test_atoms,
                max_epochs=full_max_epochs,
                learning_rate=full_lr,
                **common,
            )
            base_path = (run_dir / "base_model.pt").resolve()
            full_path = (run_dir / "full_model.pt").resolve()
            torch.save(base_run["model"], base_path)
            torch.save(full_run["model"], full_path)
            artifacts = {}
            for prefix, run in (("base", base_run), ("full", full_run)):
                artifacts[f"{prefix}_loss_plot"] = _save_loss_plot(
                    run_dir,
                    run["history"],
                    title=f"{prefix.title()} model",
                    filename=f"{prefix}_loss.png",
                )
                artifacts[f"{prefix}_validation_mae_plot"] = (
                    _save_epoch_mae_plot(
                        run_dir,
                        run["history"],
                        title=f"{prefix.title()} model validation MAE",
                        filename=f"{prefix}_validation_mae.png",
                    )
                )
            artifacts.update(
                _save_scratch_mae_plot(
                    run_dir,
                    base_run["metrics"],
                    full_run["metrics"],
                    cross_validation=False,
                )
            )
            result.update(
                {
                    "status": "completed",
                    "config": config,
                    "transfer_learning": False,
                    "cross_validation": False,
                    "warnings": warnings,
                    "seed": seed,
                    "device": str(device),
                    "dataset_sizes": {
                        "base": len(base_atoms),
                        "transfer": len(full_atoms),
                        "base_test": len(base_test_atoms),
                        "transfer_test": len(full_test_atoms),
                        "train": len(train_indices),
                        "validation": len(valid_indices),
                    },
                    "train_indices": train_indices,
                    "validation_indices": valid_indices,
                    "metrics": {
                        "base_model": base_run["metrics"],
                        "full_high_fidelity_model": full_run["metrics"],
                    },
                    "scratch_training": {
                        "base_model": {
                            key: value
                            for key, value in base_run.items()
                            if key != "model"
                        },
                        "full_high_fidelity_model": {
                            key: value
                            for key, value in full_run.items()
                            if key != "model"
                        },
                    },
                    "models": {
                        "base_model": str(base_path),
                        "full_high_fidelity_model": str(full_path),
                    },
                    "artifacts": artifacts,
                }
            )

        _write_json(result_path, result)
        return result_path
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        _write_json(result_path, result)
        raise


def _is_scratch_config(path: Path) -> bool:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(config, dict) and config.get("transfer_learning") is False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    configs = [
        path
        for path in sorted(args.input_dir.resolve().glob("*.json"))
        if _is_scratch_config(path)
    ]
    if not configs:
        print("No scratch-training JSON input files found", file=sys.stderr)
        return 1
    failed = 0
    for config_path in configs:
        print(f"\nRunning scratch configuration: {config_path}")
        try:
            print(f"Results written to {run_config(config_path, args.output_dir.resolve())}")
        except Exception as exc:
            failed += 1
            print(f"Run failed: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
