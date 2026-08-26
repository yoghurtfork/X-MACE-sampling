"""Train one scratch X-MACE model from a JSON configuration."""

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
    _checkpoint_epochs_from_config, _e0s_from_config, _e0s_from_metadata, _import_project_modules, _is_scratch_config,
    _next_run_dir, _path, _read_atoms, _required, _save_epoch_mae_plot,
    _save_fold_selection_plot, _save_loss_plot, _train_k_fold_models,
    _train_model, _strategy_from_config, _strategy_kwargs_from_config, _trainer_options_from_config, _validate_device, _write_json,
    seed_everything,
)


def _stage_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the input and output settings for the requested scratch stage."""
    run = config.get("run")
    if run not in {"just_base", "just_full"}:
        raise ValueError("single_trainer.py requires 'run' to be 'just_base' or 'just_full'")
    if run == "just_base":
        return {
            "run": run,
            "label": "base",
            "size_key": "base",
            "model_key": "base_model",
            "cv_model_key": "base_models",
            "model_prefix": "base_model",
            "xyz_key": "base_xyz",
            "test_xyz_key": "base_test_xyz",
            "count_key": "base_n_geometries",
            "e0_key": "base_E0s",
            "epochs_key": "base_max_epochs",
            "learning_rate_key": "base_learning_rate",
        }
    return {
        "run": run,
        "label": "full high-fidelity",
        "size_key": "transfer",
        "model_key": "full_high_fidelity_model",
        "cv_model_key": "full_high_fidelity_models",
        "model_prefix": "full_model",
        "xyz_key": "transfer_xyz",
        "test_xyz_key": "transfer_test_xyz",
        "count_key": "transfer_n_geometries",
        "e0_key": "full_E0s",
        "epochs_key": "full_max_epochs",
        "learning_rate_key": "full_learning_rate",
    }


def run_config(config_path: Path, output_dir: Path, run_dir: Path | None = None) -> Path:
    (
        modules, AtomDataLoaderBuilder, Tester, _extract_latent_space,
        _NaiveStrategy, Trainer, initialise_autoencoder, descriptors, _selectors,
    ) = _import_project_modules()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The top-level JSON value must be an object")
    if config.get("transfer_learning", True) is not False:
        raise ValueError("single_trainer.py requires 'transfer_learning': false")
    stage = _stage_config(config)

    run_dir = _next_run_dir(output_dir) if run_dir is None else run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Preallocated run directory does not exist: {run_dir}")
    result_path = run_dir / "result.json"
    result: dict[str, Any] = {
        "status": "running", "input_file": str(config_path.resolve()),
        "run_directory": str(run_dir), "program": "single_trainer.py",
        "run": stage["run"],
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
            k = config.get("k")
            if isinstance(k, bool) or not isinstance(k, int) or k < 2:
                raise ValueError("'k' must be a JSON integer of at least 2 when cross-validation is true")
        else:
            k = None
        max_epochs = int(config.get("max_epochs", MAX_EPOCHS))
        stage_epochs = int(config.get(stage["epochs_key"], max_epochs))
        learning_rate = float(config.get(stage["learning_rate_key"], SCRATCH_LR))
        batch_size = int(config.get("batch_size", BATCH_SIZE))
        r_max = float(config.get("r_max", R_MAX))
        validation_fraction = float(config.get("validation_fraction", VALIDATION_FRACTION))
        if min(max_epochs, stage_epochs, batch_size) < 1:
            raise ValueError("Epoch counts and batch size must be positive")
        if learning_rate <= 0:
            raise ValueError("Learning rates must be positive")
        if not cross_validation and not 0 < validation_fraction < 1:
            raise ValueError("'validation_fraction' must be between 0 and 1")
        device = _validate_device(str(config.get("device", DEVICE)))
        e0s = _e0s_from_config(config, stage["e0_key"])
        strategy = _strategy_from_config(config)
        strategy_kwargs = _strategy_kwargs_from_config(config, strategy)
        trainer_options = _trainer_options_from_config(config)
        generate_plots = trainer_options["verbose"]
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
            raise ValueError("'load_base' must be a non-empty string, true, false, or null")

        atoms = _read_atoms(_path(_required(config, stage["xyz_key"]), config_path), config.get(stage["count_key"]))
        test_atoms = _read_atoms(_path(_required(config, stage["test_xyz_key"]), config_path))
        tester = Tester(device=device)
        loss_defaults = {"energy_weight": 1.0, "forces_weight": 5.0, "dipoles_weight": 0.0, "nacs_weight": 0.0, "socs_weight": 0.0}
        loss_defaults.update(config.get("loss_kwargs", {}))
        loss_fn = modules.InvariantsWeightedEnergyForcesNacsDipoleLoss(**loss_defaults).to(device)
        energy_key = str(config.get("energy_key", "REF_energy"))
        forces_key = str(config.get("forces_key", "REF_forces"))

        if cross_validation:
            builder = AtomDataLoaderBuilder(cutoff=r_max, energy_key=energy_key, forces_key=forces_key, E0s=e0s)
            builder.load(atoms, batch_size=batch_size, shuffle=False)
            resolved_e0s = _e0s_from_metadata(builder.get_metadata())
            seed_everything(seed)
            initial_model = initialise_autoencoder(builder.get_metadata(), preset=preset, load_base=load_base).to(device)
            partial_run: dict[str, Any] = {}

            def checkpoint(snapshot: dict[str, Any]) -> None:
                partial_run.update(snapshot)
                checkpoint_result = {"cross_validation_progress": {stage["cv_model_key"]: {"completed_folds": snapshot["completed_folds"], "total_folds": snapshot["total_folds"]}}, "metrics": {stage["cv_model_key"]: snapshot["aggregate_test_metrics"]}, "cross_validation_training": {stage["cv_model_key"]: snapshot}, "models": {stage["cv_model_key"]: snapshot["model_paths"]}}
                if generate_plots:
                    checkpoint_result["artifacts"] = {stage["cv_model_key"]: snapshot["artifacts"]}
                result.update(checkpoint_result)
                _write_json(result_path, result)

            model_run = _train_k_fold_models(
                initial_model=initial_model, all_atoms=atoms, test_atoms=test_atoms,
                model_prefix=stage["model_prefix"], run_dir=run_dir,
                data_builder_class=AtomDataLoaderBuilder, trainer_class=Trainer,
                tester=tester, loss_fn=loss_fn, device=device, seed=seed, k=k,
                r_max=r_max, batch_size=batch_size, max_epochs=stage_epochs,
                learning_rate=learning_rate, trainer_options=trainer_options,
                energy_key=energy_key, forces_key=forces_key, e0s=e0s,
                checkpoint_epochs=checkpoint_epochs,
                strategy=strategy,
                strategy_kwargs=strategy_kwargs,
                generate_plots=generate_plots,
                on_fold_complete=checkpoint,
                on_checkpoint=checkpoint,
            )
            if generate_plots:
                splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
                for fold_number, (train_indices, valid_indices) in enumerate(splitter.split(range(len(atoms))), start=1):
                    model_run["artifacts"][f"model_{fold_number}"]["selection_plot"] = _save_fold_selection_plot(run_dir=run_dir, base_atoms=atoms, base_test_atoms=test_atoms, sampled_global_indices=np.arange(len(atoms)), fold_train_indices=train_indices, fold_valid_indices=valid_indices, fold_number=fold_number, model_prefix=stage["model_prefix"], descriptors=descriptors)
            completed_result = {"status": "completed", "config": config, "transfer_learning": False, "cross_validation": True, "k": k, "seed": seed, "device": str(device), "E0s": {stage["size_key"]: resolved_e0s}, "trainer_options": trainer_options, "model_initialization": {"preset": preset, "load_base": load_base}, "dataset_sizes": {stage["size_key"]: len(atoms), f"{stage['size_key']}_test": len(test_atoms)}, "metrics": {stage["cv_model_key"]: model_run["aggregate_test_metrics"]}, "cross_validation_training": {stage["cv_model_key"]: model_run}, "cross_validation_progress": {stage["cv_model_key"]: {"completed_folds": model_run["completed_folds"], "total_folds": model_run["total_folds"]}}, "models": {stage["cv_model_key"]: model_run["model_paths"]}}
            if generate_plots:
                completed_result["artifacts"] = {stage["cv_model_key"]: model_run["artifacts"]}
            result.update(completed_result)
        else:
            indices = np.arange(len(atoms))
            train_indices, valid_indices = train_test_split(indices, test_size=validation_fraction, random_state=seed, shuffle=True)
            model_path = (run_dir / f"{stage['model_prefix']}.pt").resolve()

            def checkpoint_model(snapshot: dict[str, Any]) -> None:
                result.update({
                    "models": {stage["model_key"]: snapshot["model_path"]},
                    "scratch_training": {stage["model_key"]: snapshot},
                })
                _write_json(result_path, result)

            model_run = _train_model(train_atoms=[atoms[i] for i in train_indices], valid_atoms=[atoms[i] for i in valid_indices], test_atoms=test_atoms, builder_class=AtomDataLoaderBuilder, trainer_class=Trainer, initialise_autoencoder=initialise_autoencoder, tester=tester, loss_fn=loss_fn, device=device, seed=seed, r_max=r_max, batch_size=batch_size, max_epochs=stage_epochs, learning_rate=learning_rate, trainer_options=trainer_options, energy_key=energy_key, forces_key=forces_key, preset=preset, load_base=load_base, e0s=e0s, checkpoint_epochs=checkpoint_epochs, strategy=strategy, strategy_kwargs=strategy_kwargs, model_path=model_path, on_checkpoint=checkpoint_model)
            completed_result = {"status": "completed", "config": config, "transfer_learning": False, "cross_validation": False, "seed": seed, "device": str(device), "E0s": {stage["size_key"]: model_run["E0s"]}, "trainer_options": trainer_options, "model_initialization": {"preset": preset, "load_base": load_base}, "dataset_sizes": {stage["size_key"]: len(atoms), f"{stage['size_key']}_test": len(test_atoms), "train": len(train_indices), "validation": len(valid_indices)}, "train_indices": train_indices, "validation_indices": valid_indices, "metrics": {stage["model_key"]: model_run["metrics"]}, "scratch_training": {stage["model_key"]: {key: value for key, value in model_run.items() if key != "model"}}, "models": {stage["model_key"]: str(model_path)}}
            if generate_plots:
                completed_result["artifacts"] = {"loss_plot": _save_loss_plot(run_dir, model_run["history"], title=f"{stage['label'].title()} model", filename=f"{stage['model_prefix']}_loss.png"), "validation_mae_plot": _save_epoch_mae_plot(run_dir, model_run["history"], title=f"{stage['label'].title()} model validation MAE", filename=f"{stage['model_prefix']}_validation_mae.png")}
            result.update(completed_result)
        _write_json(result_path, result)
        return result_path
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        _write_json(result_path, result)
        raise
    except Exception as exc:
        result.update({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}})
        _write_json(result_path, result)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    configs = [path for path in sorted(args.input_dir.resolve().glob("*.json")) if _is_scratch_config(path)]
    configs = [path for path in configs if json.loads(path.read_text(encoding="utf-8")).get("run") in {"just_base", "just_full"}]
    if not configs:
        print("No single-stage scratch-training JSON input files found", file=sys.stderr)
        return 1
    failed = 0
    for config_path in configs:
        try:
            print(f"Results written to {run_config(config_path, args.output_dir.resolve())}")
        except Exception as exc:
            failed += 1
            print(f"Run failed: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
