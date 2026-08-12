"""Run JSON-configured X-MACE transfer-learning experiments.

Each ``*.json`` file in ``scripts/input`` is treated as one independent run.
Results are written to the next available ``scripts/output/run_<index>`` folder.
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
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.helper import (
    BATCH_SIZE, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, DEVICE,
    MAX_EPOCHS, R_MAX, SEED, TRANSFER_LR, VALIDATION_FRACTION,
    _e0s_from_config, _e0s_from_metadata, _evaluate,
    _import_project_modules, _load_model, _next_run_dir, _path, _read_atoms,
    _required, _save_epoch_mae_plot, _save_fold_selection_plot,
    _save_loss_plot, _save_mae_plot, _save_pca_selection_plots,
    _save_selection_plot, _save_split_plot, _train_k_fold_models,
    _save_model,
    _trainer_options_for_learning_rate, _trainer_options_from_config,
    _validate_device, _write_json, seed_everything,
)


def run_config(
    config_path: Path, output_dir: Path, run_dir: Path | None = None
) -> Path:
    """Execute one input configuration and return its result JSON path."""
    (
        modules,
        AtomDataLoaderBuilder,
        Tester,
        extract_latent_space,
        NaiveStrategy,
        Trainer,
        initialise_autoencoder,
        descriptors,
        selectors,
    ) = _import_project_modules()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The top-level JSON value must be an object")
    transfer_learning = config.get("transfer_learning", True)
    if not isinstance(transfer_learning, bool):
        raise ValueError("'transfer_learning' must be a JSON boolean")
    if not transfer_learning:
        raise ValueError(
            "tester.py only runs transfer-learning configurations; "
            "use base_full_trainer.py when 'transfer_learning' is false"
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
    }
    _write_json(result_path, result)

    try:
        result["config"] = config
        _write_json(result_path, result)
        seed = int(config.get("seed", SEED))
        cross_validation_value = config.get("cross_validation", False)
        if not isinstance(cross_validation_value, bool):
            raise ValueError("'cross_validation' must be a JSON boolean")
        cross_validation = cross_validation_value
        if cross_validation:
            if "k" not in config:
                raise ValueError(
                    "'k' is required when 'cross_validation' is true"
                )
            if isinstance(config["k"], bool) or not isinstance(
                config["k"], int
            ):
                raise ValueError("'k' must be a JSON integer")
            k = config["k"]
            if k < 2:
                raise ValueError("'k' must be at least 2")
        else:
            k = None
        max_epochs = int(config.get("max_epochs", MAX_EPOCHS))
        r_max = float(config.get("r_max", R_MAX))
        batch_size = int(config.get("batch_size", BATCH_SIZE))
        transfer_lr = float(config.get("transfer_lr", TRANSFER_LR))
        validation_fraction = float(
            config.get("validation_fraction", VALIDATION_FRACTION)
        )
        device = _validate_device(str(config.get("device", DEVICE)))
        base_e0s = _e0s_from_config(config, "base_E0s")
        full_e0s = _e0s_from_config(config, "full_E0s")
        trainer_options = _trainer_options_from_config(config)

        if not cross_validation and not 0.0 < validation_fraction < 1.0:
            raise ValueError("'validation_fraction' must be between 0 and 1")
        if max_epochs < 1 or batch_size < 1:
            raise ValueError("'max_epochs' and 'batch_size' must be positive")

        seed_everything(seed)

        base_xyz = _path(_required(config, "base_xyz"), config_path)
        transfer_xyz = _path(_required(config, "transfer_xyz"), config_path)
        base_test_xyz = _path(_required(config, "base_test_xyz"), config_path)
        transfer_test_xyz = _path(
            _required(config, "transfer_test_xyz"), config_path
        )
        base_atoms = _read_atoms(base_xyz, config.get("base_n_geometries"))
        transfer_atoms = _read_atoms(
            transfer_xyz, config.get("transfer_n_geometries")
        )
        base_test_atoms = _read_atoms(base_test_xyz)
        transfer_test_atoms = _read_atoms(transfer_test_xyz)
        if len(base_atoms) != len(transfer_atoms):
            raise ValueError(
                "Base and transfer datasets must contain the same number "
                "of aligned geometries"
            )
        if len(base_atoms) < 2:
            raise ValueError("At least two grid geometries are required")

        all_indices = np.arange(len(base_atoms))
        warnings = []
        if cross_validation:
            train_indices = all_indices
            valid_indices = np.asarray([], dtype=int)
            if "validation_fraction" in config:
                warnings.append(
                    "Ignoring 'validation_fraction' because "
                    "cross-validation replaces the fixed validation split"
                )
        else:
            train_indices, valid_indices = train_test_split(
                all_indices,
                test_size=validation_fraction,
                random_state=seed,
                shuffle=True,
            )
            if "k" in config:
                warnings.append(
                    "Ignoring 'k' because cross-validation is disabled"
                )
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        base_train_atoms = [base_atoms[i] for i in train_indices]
        transfer_train_atoms = [transfer_atoms[i] for i in train_indices]
        transfer_valid_atoms = [transfer_atoms[i] for i in valid_indices]

        base_data_builder = AtomDataLoaderBuilder(
            cutoff=r_max,
            energy_key=str(config.get("energy_key", "REF_energy")),
            forces_key=str(config.get("forces_key", "REF_forces")),
            E0s=base_e0s,
        )
        full_data_builder = AtomDataLoaderBuilder(
            cutoff=r_max,
            energy_key=str(config.get("energy_key", "REF_energy")),
            forces_key=str(config.get("forces_key", "REF_forces")),
            E0s=full_e0s,
        )
        transfer_trainer_options = _trainer_options_for_learning_rate(
            trainer_options, transfer_lr
        )
        trainer = Trainer(
            max_epochs=max_epochs,
            device=device,
            **transfer_trainer_options,
        )
        tester = Tester(device=device)
        loss_kwargs = {
            "energy_weight": 1.0,
            "forces_weight": 5.0,
            "dipoles_weight": 0.0,
            "nacs_weight": 0.0,
            "socs_weight": 0.0,
        }
        loss_kwargs.update(config.get("loss_kwargs", {}))
        loss_fn = modules.InvariantsWeightedEnergyForcesNacsDipoleLoss(
            **loss_kwargs
        ).to(device)

        base_model_path = _path(
            _required(config, "base_model_path"), config_path
        )
        full_model_path = _path(
            _required(config, "full_model_path"), config_path
        )
        split_plot = None
        if not cross_validation:
            split_plot = _save_split_plot(
                run_dir,
                base_atoms,
                base_test_atoms,
                train_indices,
                valid_indices,
                descriptors,
            )
        # Build metadata, including automatic E0s, from the training datasets
        # before evaluating their held-out test sets.
        base_data_builder.load(base_atoms, batch_size=batch_size, shuffle=False)
        full_data_builder.load(
            transfer_atoms, batch_size=batch_size, shuffle=False
        )
        resolved_e0s = {
            "base": _e0s_from_metadata(base_data_builder.get_metadata()),
            "full": _e0s_from_metadata(full_data_builder.get_metadata()),
        }
        base_test_loader = base_data_builder.load(
            base_test_atoms, batch_size=batch_size, shuffle=False
        )
        transfer_test_loader = full_data_builder.load(
            transfer_test_atoms, batch_size=batch_size, shuffle=False
        )
        base_model = _load_model(base_model_path, device)
        full_model = _load_model(full_model_path, device)
        base_metrics = _evaluate(base_model, base_test_loader, tester)
        full_metrics = _evaluate(full_model, transfer_test_loader, tester)

        descriptor_name = str(_required(config, "descriptor"))
        descriptor_kwargs = dict(config.get("descriptor_kwargs", {}))
        if descriptor_name == "latent_space":
            unshuffled_loader = base_data_builder.load(
                base_train_atoms, batch_size=batch_size, shuffle=False
            )
            descriptor_matrix = extract_latent_space(
                base_model, unshuffled_loader, device=device
            )
        elif descriptor_name == "encoded_energies":
            encoded_rows = []
            base_model.eval()
            with torch.no_grad():
                for atom in base_train_atoms:
                    energies = descriptors.get_descriptor("energies", atom)
                    energy_tensor = torch.as_tensor(
                        energies,
                        dtype=torch.get_default_dtype(),
                        device=device,
                    ).reshape(1, -1, 1)
                    encoded_rows.append(
                        base_model.perm_encoder(energy_tensor)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .tolist()
                    )
            descriptor_matrix = np.asarray(encoded_rows)
        else:
            descriptor_matrix = np.asarray(
                [
                    descriptors.get_descriptor(
                        descriptor_name,
                        atom,
                        encoder=base_model.perm_encoder,
                        **descriptor_kwargs,
                    )
                    for atom in base_train_atoms
                ]
            )
        descriptor_matrix = np.asarray(descriptor_matrix)
        if descriptor_matrix.ndim == 1:
            descriptor_matrix = descriptor_matrix.reshape(-1, 1)
        if descriptor_matrix.ndim != 2:
            raise ValueError(
                "Descriptor matrix must be two-dimensional; "
                f"got shape {descriptor_matrix.shape}"
            )
        raw_descriptor_shape = list(descriptor_matrix.shape)

        pca_config = config.get("pca", False)
        pca_shape = None
        if pca_config:
            pca_kwargs = {} if pca_config is True else dict(pca_config)
            descriptor_matrix = StandardScaler().fit_transform(
                descriptor_matrix
            )
            descriptor_matrix = PCA(**pca_kwargs).fit_transform(
                descriptor_matrix
            )
            pca_shape = list(descriptor_matrix.shape)

        selector_name = str(_required(config, "selector"))
        selector_kwargs = dict(config.get("selector_kwargs", {}))
        if selector_name.startswith("k_means"):
            selector_kwargs.setdefault("random_state", seed)
        n_samples = int(_required(config, "n_samples"))
        if not 1 <= n_samples <= len(transfer_train_atoms):
            raise ValueError(
                "'n_samples' must be between 1 and the training-pool size"
            )
        # Reset immediately before selection so it is independent of model loading.
        seed_everything(seed)
        sampled_indices = np.asarray(
            selectors.get_selector(
                selector_name,
                descriptor_matrix,
                n_samples,
                **selector_kwargs,
            ),
            dtype=int,
        )
        if len(sampled_indices) != n_samples:
            raise RuntimeError(
                f"Selector returned {len(sampled_indices)} indices; "
                f"expected {n_samples}"
            )
        if len(np.unique(sampled_indices)) != len(sampled_indices):
            raise RuntimeError("Selector returned duplicate indices")
        sampled_atoms = [transfer_train_atoms[i] for i in sampled_indices]
        selection_plot = _save_selection_plot(
            run_dir,
            base_atoms,
            base_test_atoms,
            train_indices,
            valid_indices,
            sampled_indices,
            descriptors,
        )
        pca_plots = _save_pca_selection_plots(
            run_dir,
            descriptor_matrix,
            sampled_indices,
            descriptor_name,
            selector_name,
        )

        if cross_validation:
            transfer_initial_model = NaiveStrategy().apply(base_model)

            def checkpoint_transfer_fold(snapshot: dict[str, Any]) -> None:
                result.update(
                    {
                        "status": "running",
                        "cross_validation_progress": {
                            "transfer_models": {
                                "completed_folds": snapshot["completed_folds"],
                                "total_folds": snapshot["total_folds"],
                            }
                        },
                        "metrics": {
                            "base_model": base_metrics,
                            "full_high_fidelity_model": full_metrics,
                            "transfer_models": snapshot[
                                "aggregate_test_metrics"
                            ],
                        },
                        "cross_validation_training": {
                            "transfer_models": snapshot
                        },
                        "models": {
                            "transfer_models": snapshot["model_paths"]
                        },
                        "artifacts": {
                            "data_split_plot": split_plot,
                            "sample_selection_plot": selection_plot,
                            **pca_plots,
                            "transfer_models": snapshot["artifacts"]
                        },
                    }
                )
                _write_json(result_path, result)

            transfer_cv = _train_k_fold_models(
                initial_model=transfer_initial_model,
                all_atoms=sampled_atoms,
                test_atoms=transfer_test_atoms,
                model_prefix="transfer_model",
                run_dir=run_dir,
                data_builder_class=AtomDataLoaderBuilder,
                trainer_class=Trainer,
                tester=tester,
                loss_fn=loss_fn,
                device=device,
                seed=seed,
                k=k,
                r_max=r_max,
                batch_size=batch_size,
                max_epochs=max_epochs,
                learning_rate=transfer_lr,
                trainer_options=trainer_options,
                energy_key=str(config.get("energy_key", "REF_energy")),
                forces_key=str(config.get("forces_key", "REF_forces")),
                e0s=full_e0s,
                on_fold_complete=checkpoint_transfer_fold,
                on_checkpoint=checkpoint_transfer_fold,
            )
            sampled_global_indices = train_indices[sampled_indices]
            fold_selection_plots = {}
            splitter = KFold(
                n_splits=k, shuffle=True, random_state=seed
            )
            for fold_number, (
                fold_train_indices,
                fold_valid_indices,
            ) in enumerate(
                splitter.split(range(len(sampled_atoms))), start=1
            ):
                model_key = f"model_{fold_number}"
                fold_selection_plots[model_key] = (
                    _save_fold_selection_plot(
                        run_dir=run_dir,
                        base_atoms=base_atoms,
                        base_test_atoms=base_test_atoms,
                        sampled_global_indices=sampled_global_indices,
                        fold_train_indices=fold_train_indices,
                        fold_valid_indices=fold_valid_indices,
                        fold_number=fold_number,
                        model_prefix="transfer_model",
                        descriptors=descriptors,
                    )
                )
                transfer_cv["artifacts"][model_key][
                    "selection_plot"
                ] = fold_selection_plots[model_key]

            metrics_plot = _save_mae_plot(
                run_dir,
                base_metrics,
                full_metrics,
                transfer_cv["aggregate_test_metrics"],
                transfer_cv["combined_validation"]["best_epoch"],
                cross_validation=True,
            )
            result.update(
                {
                    "status": "completed",
                    "config": config,
                    "transfer_learning": True,
                    "cross_validation": True,
                    "k": k,
                    "warnings": warnings,
                    "seed": seed,
                    "device": str(device),
                    "E0s": resolved_e0s,
                    "trainer_options": trainer_options,
                    "dataset_sizes": {
                        "base": len(base_atoms),
                        "transfer": len(transfer_atoms),
                        "base_test": len(base_test_atoms),
                        "transfer_test": len(transfer_test_atoms),
                        "selection_pool": len(train_indices),
                        "sampled": len(sampled_indices),
                    },
                    "sampled_indices": sampled_indices,
                    "descriptor": {
                        "name": descriptor_name,
                        "kwargs": descriptor_kwargs,
                        "shape": raw_descriptor_shape,
                        "pca_shape": pca_shape,
                    },
                    "selector": {
                        "name": selector_name,
                        "kwargs": selector_kwargs,
                        "n_samples": n_samples,
                    },
                    "metrics": {
                        "base_model": base_metrics,
                        "full_high_fidelity_model": full_metrics,
                        "transfer_models": transfer_cv[
                            "aggregate_test_metrics"
                        ],
                    },
                    "cross_validation_training": {
                        "transfer_models": {
                            "folds": transfer_cv["folds"],
                            "combined_validation": transfer_cv[
                                "combined_validation"
                            ],
                            "aggregate_test_metrics": transfer_cv[
                                "aggregate_test_metrics"
                            ],
                            "training_seconds": transfer_cv[
                                "training_seconds"
                            ],
                        }
                    },
                    "cross_validation_progress": {
                        "transfer_models": {
                            "completed_folds": transfer_cv["completed_folds"],
                            "total_folds": transfer_cv["total_folds"],
                        }
                    },
                    "models": {
                        "transfer_models": transfer_cv["model_paths"]
                    },
                    "artifacts": {
                        "sample_selection_plot": selection_plot,
                        **pca_plots,
                        "transfer_models": transfer_cv["artifacts"],
                        "final_metrics_comparison_plot": metrics_plot,
                    },
                }
            )
            _write_json(result_path, result)
            return result_path

        transfer_train_loader = full_data_builder.load(
            sampled_atoms, batch_size=batch_size, shuffle=True
        )
        transfer_valid_loader = full_data_builder.load(
            transfer_valid_atoms, batch_size=batch_size, shuffle=False
        )
        transfer_model = NaiveStrategy().apply(base_model)
        transfer_model_path = (run_dir / "transfer_model.pt").resolve()
        seed_everything(seed)
        started_at = time.time()
        try:
            transfer_model, transfer_history = trainer.train_model(
                transfer_model,
                transfer_train_loader,
                transfer_valid_loader,
                loss_fn,
            )
        except KeyboardInterrupt:
            _save_model(transfer_model, transfer_model_path)
            result.update({
                "models": {"transfer_model": str(transfer_model_path)},
                "transfer_training": {
                    "status": "interrupted",
                    "model_path": str(transfer_model_path),
                    "training_seconds": time.time() - started_at,
                },
            })
            _write_json(result_path, result)
            raise
        _save_model(transfer_model, transfer_model_path)
        training_seconds = time.time() - started_at
        result.update({
            "models": {"transfer_model": str(transfer_model_path)},
            "transfer_training": {
                "status": "trained_pending_evaluation",
                "model_path": str(transfer_model_path),
                "best_epoch": transfer_history["best_epoch"],
                "training_seconds": training_seconds,
                "history": transfer_history,
            },
        })
        _write_json(result_path, result)
        transfer_metrics = _evaluate(
            transfer_model, transfer_test_loader, tester
        )
        transfer_metrics["best_epoch"] = int(transfer_history["best_epoch"])
        result.update({
            "metrics": {
                "base_model": base_metrics,
                "full_high_fidelity_model": full_metrics,
                "transfer_model": transfer_metrics,
            },
            "transfer_training": {
                "status": "evaluated_pending_artifacts",
                "model_path": str(transfer_model_path),
                "best_epoch": transfer_history["best_epoch"],
                "training_seconds": training_seconds,
                "history": transfer_history,
            },
        })
        _write_json(result_path, result)

        loss_plot = _save_loss_plot(run_dir, transfer_history)
        metrics_plot = _save_mae_plot(
            run_dir,
            base_metrics,
            full_metrics,
            transfer_metrics,
            transfer_history["best_epoch"],
            cross_validation=False,
        )
        result.update(
            {
                "status": "completed",
                "config": config,
                "transfer_learning": True,
                "cross_validation": False,
                "warnings": warnings,
                "seed": seed,
                "device": str(device),
                "E0s": resolved_e0s,
                "trainer_options": trainer_options,
                "dataset_sizes": {
                    "base": len(base_atoms),
                    "transfer": len(transfer_atoms),
                    "base_test": len(base_test_atoms),
                    "transfer_test": len(transfer_test_atoms),
                    "train": len(train_indices),
                    "validation": len(valid_indices),
                    "sampled": len(sampled_indices),
                },
                "train_indices": train_indices,
                "validation_indices": valid_indices,
                "sampled_indices": sampled_indices,
                "descriptor": {
                    "name": descriptor_name,
                    "kwargs": descriptor_kwargs,
                    "shape": raw_descriptor_shape,
                    "pca_shape": pca_shape,
                },
                "selector": {
                    "name": selector_name,
                    "kwargs": selector_kwargs,
                    "n_samples": n_samples,
                },
                "metrics": {
                    "base_model": base_metrics,
                    "full_high_fidelity_model": full_metrics,
                    "transfer_model": transfer_metrics,
                },
                "transfer_training": {
                    "status": "completed",
                    "model_path": str(transfer_model_path),
                    "best_epoch": transfer_history["best_epoch"],
                    "training_seconds": training_seconds,
                    "history": transfer_history,
                },
                "models": {"transfer_model": str(transfer_model_path)},
                "artifacts": {
                    "data_split_plot": split_plot,
                    "sample_selection_plot": selection_plot,
                    **pca_plots,
                    "transfer_loss_plot": loss_plot,
                    "final_metrics_comparison_plot": metrics_plot,
                },
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
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="directory containing run configuration JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory in which run_<index> folders are created",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    config_paths = []
    for config_path in sorted(input_dir.glob("*.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(config, dict) and config.get("transfer_learning", True):
            config_paths.append(config_path)
    if not config_paths:
        print(
            f"No transfer-learning JSON input files found in {input_dir}",
            file=sys.stderr,
        )
        return 1

    failed = 0
    for config_path in config_paths:
        print(f"\nRunning configuration: {config_path}")
        try:
            result_path = run_config(config_path, output_dir)
            print(f"Results written to {result_path}")
        except Exception as exc:
            failed += 1
            print(f"Run failed: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
