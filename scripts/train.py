"""The single CLI and programmatic entry point for X-MACE training."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from scripts import data, evaluation, model, state
from scripts.config import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def run_config(
    config_path: Path, output_dir: Path, run_dir: Path | None = None
) -> Path:
    """Run one normalized configuration and return its ``result.json`` path."""
    config_path = Path(config_path).resolve()
    config, warnings = load_config(config_path)
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if run_dir is None:
        run_dir = state.reserve_run_dir(output_dir)
    else:
        run_dir = Path(run_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Reserved run directory does not exist: {run_dir}")
    run_state = state.RunState(run_dir, config_path, config, warnings)
    try:
        device = model.validate_device(config["device"])
        if config["mode"] in {"lf", "both"}:
            _run_scratch_stage("lf", config, device, run_state)
        if config["mode"] in {"hf", "both"}:
            _run_scratch_stage("hf", config, device, run_state)
        if config["mode"] == "transfer":
            _run_transfer_stage(config, device, run_state)
    except KeyboardInterrupt:
        run_state.interrupt("Training interrupted")
        raise
    except Exception as error:
        run_state.fail(error)
        raise
    return run_state.complete()


def _run_scratch_stage(
    stage_name: str, config: dict[str, Any], device: Any, run_state: state.RunState
) -> None:
    stage_data = data.load_stage_data(config, stage_name)
    run_state.update_stage(stage_name, {
        "status": "running",
        "dataset_sizes": {
            "training": len(stage_data.atoms),
            "test_sets": {name: len(atoms) for name, atoms in stage_data.test_sets.items()},
        },
    })
    prefix = f"{stage_name}_model"
    if config["cross_validation"]:
        initial_model = _scratch_initial_model(stage_data.builder, config, device)
        fold_results: dict[str, dict[str, Any]] = {}
        for fold_number, (train_indices, valid_indices) in enumerate(
            data.kfold_splits(len(stage_data.atoms), config["k"], config["seed"]), start=1
        ):
            result = _train_once(
                initial_model=deepcopy(initial_model), stage_name=stage_name,
                builder=stage_data.builder,
                train_atoms=_atoms_at(stage_data.atoms, train_indices),
                valid_atoms=_atoms_at(stage_data.atoms, valid_indices),
                test_sets=stage_data.test_sets, config=config, device=device,
                run_state=run_state, model_filename=f"{prefix}_fold_{fold_number}.pt",
                seed=config["seed"] + fold_number,
            )
            result["fold_seed"] = config["seed"] + fold_number
            result["train_indices"] = train_indices
            result["validation_indices"] = valid_indices
            if config["generate_plots"]:
                result["artifacts"].update(_fold_selection_artifact(
                    run_state.run_dir, stage_data.atoms, valid_indices, prefix, fold_number
                ))
            fold_results[f"fold_{fold_number}"] = result
            run_state.update_stage(stage_name, {
                "status": "running", "folds": fold_results,
                "completed_folds": len(fold_results), "total_folds": config["k"],
            })
        aggregate = _aggregate_test_sets(fold_results)
        run_state.update_stage(stage_name, {
            "status": "completed", "folds": fold_results,
            "completed_folds": len(fold_results), "total_folds": config["k"],
            "aggregate_test_metrics": aggregate,
            **_runtime_e0(stage_name, config, stage_data.resolved_e0s),
        })
        return

    train_indices, valid_indices = data.fixed_split(
        len(stage_data.atoms), config["validation_fraction"], config["seed"]
    )
    # Fixed-split fitting continues to use only the training partition, as the
    # original scratch trainer did.
    builder = data.make_builder(config, stage_name)
    train_atoms = _atoms_at(stage_data.atoms, train_indices)
    builder.load(train_atoms, batch_size=config["batch_size"], shuffle=False)
    result = _train_once(
        initial_model=_scratch_initial_model(builder, config, device), stage_name=stage_name,
        builder=builder, train_atoms=train_atoms,
        valid_atoms=_atoms_at(stage_data.atoms, valid_indices), test_sets=stage_data.test_sets,
        config=config, device=device, run_state=run_state, model_filename=f"{prefix}.pt",
        seed=config["seed"],
    )
    result.update({"train_indices": train_indices, "validation_indices": valid_indices})
    run_state.update_stage(stage_name, {
        "status": "completed", "dataset_sizes": {
            "training": len(stage_data.atoms), "train": len(train_indices),
            "validation": len(valid_indices),
            "test_sets": {name: len(atoms) for name, atoms in stage_data.test_sets.items()},
        },
        **result, **_runtime_e0(stage_name, config, data.resolved_e0s(builder)),
    })


def _run_transfer_stage(config: dict[str, Any], device: Any, run_state: state.RunState) -> None:
    lf_atoms = data.read_atoms(config["lf_xyz"], config.get("lf_n_geometries"))
    hf_atoms = data.read_atoms(config["hf_xyz"], config.get("hf_n_geometries"))
    data.validate_transfer_alignment(lf_atoms, hf_atoms)
    hf_test_sets = data.read_test_sets(config["hf_test_xyz"])
    if config["cross_validation"]:
        pool_indices = np.arange(len(lf_atoms), dtype=int)
        valid_indices = np.asarray([], dtype=int)
    else:
        pool_indices, valid_indices = data.fixed_split(
            len(lf_atoms), config["validation_fraction"], config["seed"]
        )
    lf_pool = _atoms_at(lf_atoms, pool_indices)
    hf_pool = _atoms_at(hf_atoms, pool_indices)
    initial_model = model.load_model(config["pretrained_model_path"], device)
    descriptor_matrix = _selection_descriptors(initial_model, lf_pool, config, device)
    selected_local_indices = _select_indices(descriptor_matrix, config)
    _, selected_hf_atoms, selected_local_indices = data.selected_transfer_atoms(
        lf_pool, hf_pool, selected_local_indices
    )
    selected_global_indices = pool_indices[selected_local_indices]
    artifacts: dict[str, Any] = {}
    if config["generate_plots"]:
        artifacts["selection_plot"] = state.save_pca_selection_plot(
            run_state.run_dir, descriptor_matrix, selected_local_indices,
            filename="transfer_selection_pca.png", title="LF descriptor selection",
        )
    run_state.update_stage("transfer", {
        "status": "running", "dataset_sizes": {
            "lf_training": len(lf_atoms), "hf_training": len(hf_atoms),
            "selection_pool": len(lf_pool), "selected": len(selected_hf_atoms),
            "hf_test_sets": {name: len(atoms) for name, atoms in hf_test_sets.items()},
        }, "selected_indices": selected_global_indices, "artifacts": artifacts,
    })
    hf_builder = data.make_builder(config, "hf")
    hf_builder.load(hf_atoms, batch_size=config["batch_size"], shuffle=False)
    prefix = "transfer_model"
    if config["cross_validation"]:
        fold_results: dict[str, dict[str, Any]] = {}
        for fold_number, (train_indices, fold_valid_indices) in enumerate(
            data.kfold_splits(len(selected_hf_atoms), config["k"], config["seed"]), start=1
        ):
            result = _train_once(
                initial_model=deepcopy(initial_model), stage_name="transfer", builder=hf_builder,
                train_atoms=_atoms_at(selected_hf_atoms, train_indices),
                valid_atoms=_atoms_at(selected_hf_atoms, fold_valid_indices), test_sets=hf_test_sets,
                config=config, device=device, run_state=run_state,
                model_filename=f"{prefix}_fold_{fold_number}.pt", seed=config["seed"] + fold_number,
            )
            result["fold_seed"] = config["seed"] + fold_number
            result["selected_train_indices"] = train_indices
            result["selected_validation_indices"] = fold_valid_indices
            if config["generate_plots"]:
                result["artifacts"].update(_fold_selection_artifact(
                    run_state.run_dir, selected_hf_atoms, fold_valid_indices, prefix, fold_number
                ))
            fold_results[f"fold_{fold_number}"] = result
            run_state.update_stage("transfer", {"status": "running", "folds": fold_results,
                "completed_folds": len(fold_results), "total_folds": config["k"]})
        run_state.update_stage("transfer", {
            "status": "completed", "folds": fold_results,
            "completed_folds": len(fold_results), "total_folds": config["k"],
            "aggregate_test_metrics": _aggregate_test_sets(fold_results),
            **_runtime_e0("hf", config, data.resolved_e0s(hf_builder)),
        })
        return
    result = _train_once(
        initial_model=initial_model, stage_name="transfer", builder=hf_builder,
        train_atoms=selected_hf_atoms, valid_atoms=_atoms_at(hf_atoms, valid_indices),
        test_sets=hf_test_sets, config=config, device=device, run_state=run_state,
        model_filename=f"{prefix}.pt", seed=config["seed"],
    )
    result["selection_pool_indices"] = pool_indices
    result["validation_indices"] = valid_indices
    run_state.update_stage("transfer", {
        "status": "completed", **result,
        **_runtime_e0("hf", config, data.resolved_e0s(hf_builder)),
    })


def _train_once(
    *, initial_model: Any, stage_name: str, builder: Any, train_atoms: list[Any],
    valid_atoms: list[Any], test_sets: dict[str, list[Any]], config: dict[str, Any],
    device: Any, run_state: state.RunState, model_filename: str, seed: int,
) -> dict[str, Any]:
    _seed_everything(seed)
    train_loader = data.make_loader(builder, train_atoms, config["batch_size"], shuffle=True)
    valid_loader = data.make_loader(builder, valid_atoms, config["batch_size"], shuffle=False)
    test_loaders = data.make_test_loaders(builder, test_sets, config["batch_size"])
    trained_model = model.apply_training_strategy(initial_model, config).to(device)
    started = time.monotonic()
    trained_model, history = model.build_trainer(config, stage_name, device).train_model(
        trained_model, train_loader, valid_loader, model.build_loss(config, device),
        checkpoint_epoch=config["checkpoint_epochs"],
    )
    model_path = state.save_model(trained_model, run_state.run_dir / model_filename)
    from mace.testing import Tester

    tester = Tester(device=device)
    test_metrics = evaluation.evaluate_test_sets(trained_model, test_loaders, tester)
    best_epoch = int(history["best_epoch"])
    for metrics in test_metrics.values():
        metrics["best_epoch"] = best_epoch
    checkpoint_entries = evaluation.evaluate_checkpoint_models(
        trained_model, history, config["checkpoint_epochs"], test_loaders, tester, device
    )
    checkpoints = []
    for checkpoint in checkpoint_entries:
        checkpoint_model = checkpoint.pop("model")
        checkpoint_path = model_path.with_name(
            f"{model_path.stem}_checkpoint_epoch_{checkpoint['epoch']}{model_path.suffix}"
        )
        checkpoint["model_path"] = str(state.save_model(checkpoint_model, checkpoint_path))
        checkpoints.append(checkpoint)
    safe_history = dict(history)
    safe_history["checkpoint_models"] = checkpoints
    artifacts: dict[str, str] = {}
    if config["generate_plots"]:
        stem = model_path.stem
        artifacts = {
            "loss_plot": state.save_loss_plot(run_state.run_dir, safe_history, title=stem.replace("_", " ").title(), filename=f"{stem}_loss.png"),
            "validation_mae_plot": state.save_epoch_mae_plot(run_state.run_dir, safe_history, title=f"{stem.replace('_', ' ').title()} validation MAE", filename=f"{stem}_validation_mae.png"),
        }
    return {
        "model_path": str(model_path), "checkpoint_paths": [entry["model_path"] for entry in checkpoints],
        "test_metrics": test_metrics, "metrics": test_metrics["test_1"], "best_epoch": best_epoch,
        "training_seconds": time.monotonic() - started, "history": safe_history,
        "artifacts": artifacts,
    }


def _scratch_initial_model(builder: Any, config: dict[str, Any], device: Any) -> Any:
    _seed_everything(config["seed"])
    return model.build_scratch_model(builder.get_metadata(), config, device)


def _selection_descriptors(model_instance: Any, atoms: list[Any], config: dict[str, Any], device: Any) -> np.ndarray:
    import sampling_methods.descriptors as descriptors

    name = config["descriptor"]
    kwargs = dict(config["descriptor_kwargs"])
    if name == "latent_space":
        from mace.testing import extract_latent_space
        builder = data.make_builder(config, "lf")
        loader = data.make_loader(builder, atoms, config["batch_size"], shuffle=False)
        values = extract_latent_space(model_instance, loader, device=device)
    elif name == "encoded_energies":
        import torch
        values = []
        model_instance.eval()
        with torch.no_grad():
            for atom in atoms:
                energy = descriptors.get_descriptor("energies", atom)
                tensor = torch.as_tensor(energy, dtype=torch.get_default_dtype(), device=device).reshape(1, -1, 1)
                values.append(model_instance.perm_encoder(tensor).squeeze(0).detach().cpu().tolist())
    elif name == "hessian_norm":
        values = descriptors.get_descriptor(name, atoms, **kwargs)
    else:
        values = [descriptors.get_descriptor(name, atom, encoder=model_instance.perm_encoder, **kwargs) for atom in atoms]
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError(f"Descriptor '{name}' must produce a non-empty two-dimensional matrix")
    if config["pca"]:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        dimensions = min(matrix.shape[0], matrix.shape[1])
        matrix = PCA(n_components=dimensions).fit_transform(StandardScaler().fit_transform(matrix))
    return matrix


def _select_indices(descriptor_matrix: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    import sampling_methods.selectors as selectors

    if config["n_samples"] > len(descriptor_matrix):
        raise ValueError("'n_samples' cannot exceed the transfer selection pool")
    kwargs = dict(config["selector_kwargs"])
    if config["selector"].startswith("k_means"):
        kwargs.setdefault("random_state", config["seed"])
    _seed_everything(config["seed"])
    selected = np.asarray(selectors.get_selector(config["selector"], descriptor_matrix, config["n_samples"], **kwargs), dtype=int)
    if len(selected) != config["n_samples"] or len(np.unique(selected)) != len(selected):
        raise RuntimeError("Selector must return the configured number of unique indices")
    return selected


def _aggregate_test_sets(fold_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    test_names = next(iter(fold_results.values()))["test_metrics"]
    return {
        name: evaluation.aggregate_fold_metrics({
            fold_name: {"metrics": fold["test_metrics"][name]}
            for fold_name, fold in fold_results.items()
        })
        for name in test_names
    }


def _fold_selection_artifact(run_dir: Path, atoms: list[Any], valid_indices: np.ndarray, prefix: str, fold_number: int) -> dict[str, str]:
    positions = np.asarray([np.asarray(atom.get_positions(), dtype=float).reshape(-1) for atom in atoms])
    if positions.shape[1] == 1:
        positions = np.column_stack((positions[:, 0], np.zeros(len(positions))))
    return {"selection_plot": state.save_selection_plot(
        run_dir, positions, valid_indices, filename=f"{prefix}_fold_{fold_number}_selection.png",
        title=f"{prefix.replace('_', ' ').title()} fold {fold_number} validation selection",
    )}


def _runtime_e0(stage_name: str, config: dict[str, Any], resolved: dict[str, float]) -> dict[str, Any]:
    return {} if f"{stage_name}_E0s" in config else {"resolved_E0s": resolved}


def _atoms_at(atoms: list[Any], indices: np.ndarray) -> list[Any]:
    return [atoms[int(index)] for index in indices]


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="strict training JSON configuration")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        print(run_config(args.config, args.output_dir))
    except Exception as error:
        print(f"Run failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
