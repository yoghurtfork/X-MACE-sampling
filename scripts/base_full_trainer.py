"""Train base and full high-fidelity X-MACE models from JSON configurations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.helper import (
    BATCH_SIZE, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, DEVICE,
    MAX_EPOCHS, R_MAX, SCRATCH_LR, SEED, VALIDATION_FRACTION,
    _checkpoint_epochs_from_config, _e0s_from_config, _e0s_from_metadata, _evaluate,
    _import_project_modules, _is_scratch_config, _next_run_dir, _path,
    _read_atoms, _required, _save_epoch_mae_plot, _save_fold_selection_plot,
    _save_loss_plot, _save_scratch_mae_plot, _train_k_fold_models,
    _train_model, _strategy_from_config, _strategy_kwargs_from_config, _trainer_options_from_config, _validate_device,
    _write_json, seed_everything,
)


def run_config(
    config_path: Path, output_dir: Path, run_dir: Path | None = None
) -> Path:
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
    ) = _import_project_modules()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The top-level JSON value must be an object")
    if config.get("transfer_learning", True) is not False:
        raise ValueError(
            "base_full_trainer.py requires 'transfer_learning': false"
        )

    if run_dir is None:
        run_dir = _next_run_dir(output_dir)
    else:
        run_dir = run_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"Preallocated run directory does not exist: {run_dir}"
            )
    result_path = run_dir / "result.json"
    result: dict[str, Any] = {
        "status": "running",
        "input_file": str(config_path.resolve()),
        "run_directory": str(run_dir.resolve()),
        "program": "base_full_trainer.py",
    }
    _write_json(result_path, result)

    try:
        result["config"] = config
        _write_json(result_path, result)
        seed = int(config.get("seed", SEED))
        checkpoint_epochs = _checkpoint_epochs_from_config(config)
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
        device = _validate_device(str(config.get("device", DEVICE)))
        base_e0s = _e0s_from_config(config, "base_E0s")
        full_e0s = _e0s_from_config(config, "full_E0s")
        strategy = _strategy_from_config(config)
        strategy_kwargs = _strategy_kwargs_from_config(config, strategy)
        trainer_options = _trainer_options_from_config(config)
        preset = config.get("preset", "default_ani")
        if not isinstance(preset, str) or not preset:
            raise ValueError("'preset' must be a non-empty string")
        load_base_value = config.get("load_base", "ani500k")
        if load_base_value is True:
            load_base = "ani500k"
        elif load_base_value is False or load_base_value is None:
            load_base = None
        elif isinstance(load_base_value, str) and load_base_value:
            load_base = load_base_value
        else:
            raise ValueError(
                "'load_base' must be a non-empty string, true, false, or null"
            )
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
                "trainer_options": trainer_options,
                "strategy": strategy,
                "strategy_kwargs": strategy_kwargs,
            }
            initial_models = {}
            resolved_e0s = {}
            for name, atoms, e0s in (
                ("base", base_atoms, base_e0s),
                ("full", full_atoms, full_e0s),
            ):
                builder = AtomDataLoaderBuilder(
                    cutoff=r_max,
                    energy_key=energy_key,
                    forces_key=forces_key,
                    E0s=e0s,
                )
                builder.load(atoms, batch_size=batch_size, shuffle=False)
                resolved_e0s[name] = _e0s_from_metadata(builder.get_metadata())
                seed_everything(seed)
                initial_models[name] = initialise_autoencoder(
                    builder.get_metadata(),
                    preset=preset,
                    load_base=load_base,
                ).to(device)

            partial_runs: dict[str, dict[str, Any]] = {}

            def checkpoint_fold(
                model_key: str, snapshot: dict[str, Any]
            ) -> None:
                partial_runs[model_key] = snapshot
                progress = {
                    key: {
                        "completed_folds": value["completed_folds"],
                        "total_folds": value["total_folds"],
                    }
                    for key, value in partial_runs.items()
                }
                result.update(
                    {
                        "status": "running",
                        "cross_validation_progress": progress,
                        "metrics": {
                            key: value["aggregate_test_metrics"]
                            for key, value in partial_runs.items()
                        },
                        "cross_validation_training": partial_runs,
                        "models": {
                            key: value["model_paths"]
                            for key, value in partial_runs.items()
                        },
                        "artifacts": {
                            key: value["artifacts"]
                            for key, value in partial_runs.items()
                        },
                    }
                )
                _write_json(result_path, result)

            base_run = _train_k_fold_models(
                initial_model=initial_models["base"],
                all_atoms=base_atoms,
                test_atoms=base_test_atoms,
                model_prefix="base_model",
                max_epochs=base_max_epochs,
                learning_rate=base_lr,
                e0s=base_e0s,
                checkpoint_epochs=checkpoint_epochs,
                on_fold_complete=lambda snapshot: checkpoint_fold(
                    "base_models", snapshot
                ),
                on_checkpoint=lambda snapshot: checkpoint_fold(
                    "base_models", snapshot
                ),
                **common,
            )
            full_run = _train_k_fold_models(
                initial_model=initial_models["full"],
                all_atoms=full_atoms,
                test_atoms=full_test_atoms,
                model_prefix="full_model",
                max_epochs=full_max_epochs,
                learning_rate=full_lr,
                e0s=full_e0s,
                checkpoint_epochs=checkpoint_epochs,
                on_fold_complete=lambda snapshot: checkpoint_fold(
                    "full_high_fidelity_models", snapshot
                ),
                on_checkpoint=lambda snapshot: checkpoint_fold(
                    "full_high_fidelity_models", snapshot
                ),
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
            comparison_plot = _save_scratch_mae_plot(
                run_dir,
                base_run["aggregate_test_metrics"],
                full_run["aggregate_test_metrics"],
                base_run["combined_validation"]["best_epoch"],
                full_run["combined_validation"]["best_epoch"],
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
                    "E0s": resolved_e0s,
                    "trainer_options": trainer_options,
                    "model_initialization": {
                        "preset": preset,
                        "load_base": load_base,
                    },
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
                    "cross_validation_progress": {
                        "base_models": {
                            "completed_folds": base_run["completed_folds"],
                            "total_folds": base_run["total_folds"],
                        },
                        "full_high_fidelity_models": {
                            "completed_folds": full_run["completed_folds"],
                            "total_folds": full_run["total_folds"],
                        },
                    },
                    "models": {
                        "base_models": base_run["model_paths"],
                        "full_high_fidelity_models": full_run["model_paths"],
                    },
                    "artifacts": {
                        "base_models": base_run["artifacts"],
                        "full_high_fidelity_models": full_run["artifacts"],
                        "final_metrics_comparison_plot": comparison_plot,
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
                "preset": preset,
                "load_base": load_base,
                "trainer_options": trainer_options,
            }
            base_path = (run_dir / "base_model.pt").resolve()
            full_path = (run_dir / "full_model.pt").resolve()

            def checkpoint_model(model_key: str, snapshot: dict[str, Any]) -> None:
                result.update({
                    "models": {**result.get("models", {}), model_key: snapshot["model_path"]},
                    "scratch_training": {
                        **result.get("scratch_training", {}), model_key: snapshot,
                    },
                })
                _write_json(result_path, result)

            base_run = _train_model(
                train_atoms=[base_atoms[i] for i in train_indices],
                valid_atoms=[base_atoms[i] for i in valid_indices],
                test_atoms=base_test_atoms,
                max_epochs=base_max_epochs,
                learning_rate=base_lr,
                e0s=base_e0s,
                checkpoint_epochs=checkpoint_epochs,
                model_path=base_path,
                on_checkpoint=lambda snapshot: checkpoint_model(
                    "base_model", snapshot
                ),
                **common,
            )
            full_run = _train_model(
                train_atoms=[full_atoms[i] for i in train_indices],
                valid_atoms=[full_atoms[i] for i in valid_indices],
                test_atoms=full_test_atoms,
                max_epochs=full_max_epochs,
                learning_rate=full_lr,
                e0s=full_e0s,
                checkpoint_epochs=checkpoint_epochs,
                model_path=full_path,
                on_checkpoint=lambda snapshot: checkpoint_model(
                    "full_high_fidelity_model", snapshot
                ),
                **common,
            )
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
            artifacts["final_metrics_comparison_plot"] = (
                _save_scratch_mae_plot(
                    run_dir,
                    base_run["metrics"],
                    full_run["metrics"],
                    base_run["history"]["best_epoch"],
                    full_run["history"]["best_epoch"],
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
                    "E0s": {"base": base_run["E0s"], "full": full_run["E0s"]},
                    "trainer_options": trainer_options,
                    "model_initialization": {
                        "preset": preset,
                        "load_base": load_base,
                    },
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
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        _write_json(result_path, result)
        raise
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
