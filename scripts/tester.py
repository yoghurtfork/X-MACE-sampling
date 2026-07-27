"""Run JSON-configured X-MACE sampling and transfer-learning experiments.

Each ``*.json`` file in ``scripts/input`` is treated as one independent run.
Results are written to the next available ``scripts/output/run_<index>`` folder.

Required JSON keys:
    base_xyz, transfer_xyz, base_test_xyz, transfer_test_xyz,
    base_model_path, full_model_path, descriptor, selector, n_samples

Common optional keys:
    base_n_geometries, transfer_n_geometries, descriptor_kwargs,
    selector_kwargs, seed, device, validation_fraction, batch_size, r_max,
    max_epochs, transfer_lr, patience, pca

``pca`` is either absent/false (the default), true, or an object of PCA kwargs.
Paths may be absolute or relative to the input JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import ase.io
import matplotlib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

# Notebook defaults. Every value can be overridden by an input JSON file.
SEED = 42
MAX_EPOCHS = 100
R_MAX = 5.0
BATCH_SIZE = 64
TRANSFER_LR = 5.0e-4
DEVICE = "cpu"
VALIDATION_FRACTION = 0.1
PATIENCE = 15


def _import_project_modules() -> tuple[Any, ...]:
    """Import local and X-MACE modules after making the project importable."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from mace import modules
    from mace.data.atom_data_loader import AtomDataLoaderBuilder
    from mace.testing import Tester, extract_latent_space
    from mace.training import NaiveStrategy, Trainer

    import sampling_methods.descriptors as descriptors
    import sampling_methods.selectors as selectors
    import utils.training as training

    return (
        modules,
        AtomDataLoaderBuilder,
        Tester,
        extract_latent_space,
        NaiveStrategy,
        Trainer,
        descriptors,
        selectors,
        training,
    )


def _required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required configuration key: {key!r}")
    return config[key]


def _path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _read_atoms(path: Path, limit: int | None = None) -> list[Any]:
    if not path.is_file():
        raise FileNotFoundError(f"XYZ file does not exist: {path}")
    index = ":" if limit is None else f":{int(limit)}"
    atoms = ase.io.read(path, index=index)
    return atoms if isinstance(atoms, list) else [atoms]


def _next_run_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in output_dir.glob("run_*"):
        if path.is_dir():
            try:
                indices.append(int(path.name.removeprefix("run_")))
            except ValueError:
                pass
    run_dir = output_dir / f"run_{max(indices, default=-1) + 1}"
    run_dir.mkdir()
    return run_dir


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=_json_value, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_device(name: str) -> torch.device:
    if name not in {"cpu", "cuda"}:
        raise ValueError("'device' must be either 'cpu' or 'cuda'")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _load_model(path: Path, device: torch.device) -> torch.nn.Module:
    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    model = torch.load(path, map_location=device, weights_only=False)
    return model.to(device)


def _evaluate(
    model: torch.nn.Module, loader: Any, tester: Any
) -> dict[str, float]:
    model.eval()
    tester.run_test(model, loader)
    return {
        "energy_mae_ev": float(tester.get_energy_mae()),
        "force_mae_ev_per_ang": float(tester.get_force_mae()),
    }


def _save_split_plot(
    run_dir: Path,
    base_atoms: list[Any],
    base_test_atoms: list[Any],
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    descriptors: Any,
) -> str:
    bond_lengths = [
        descriptors.get_descriptor("bond_lengths", atom)[0] for atom in base_atoms
    ]
    dihedrals = [
        descriptors.get_descriptor("dihedral", atom)[0] for atom in base_atoms
    ]
    test_bonds = [
        descriptors.get_descriptor("bond_lengths", atom)[0]
        for atom in base_test_atoms
    ]
    test_dihedrals = [
        descriptors.get_descriptor("dihedral", atom)[0]
        for atom in base_test_atoms
    ]

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.scatter(
        np.asarray(bond_lengths)[train_indices],
        np.asarray(dihedrals)[train_indices],
        color="blue",
        marker="o",
        s=10,
        alpha=0.3,
        label="Train pool",
    )
    ax.scatter(
        test_bonds,
        test_dihedrals,
        color="red",
        marker="o",
        s=10,
        label="Off-grid test set",
    )
    ax.scatter(
        np.asarray(bond_lengths)[valid_indices],
        np.asarray(dihedrals)[valid_indices],
        color="green",
        marker="o",
        s=10,
        label="Validation set",
    )
    ax.set(
        title="Test/train/validation split",
        xlabel="Bond length",
        ylabel="Dihedral angle",
    )
    ax.legend()
    fig.tight_layout()
    filename = "data_split.png"
    fig.savefig(run_dir / filename, dpi=150)
    plt.close(fig)
    return filename


def _save_loss_plot(run_dir: Path, history: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        linewidth=1.8,
        label="Train loss",
        alpha=0.8,
    )
    ax.plot(
        history["epoch"],
        history["valid_loss"],
        marker="s",
        linewidth=1.8,
        label="Validation loss",
        alpha=0.8,
    )
    ax.set(
        xlabel="Epoch",
        ylabel="Loss",
        title="Transfer learning",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    filename = "transfer_loss.png"
    fig.savefig(run_dir / filename, dpi=150)
    plt.close(fig)
    return filename


def _save_selection_plot(
    run_dir: Path,
    base_atoms: list[Any],
    base_test_atoms: list[Any],
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    sampled_indices: np.ndarray,
    descriptors: Any,
) -> str:
    """Plot the selected training geometries in bond/dihedral space."""
    bond_lengths = np.asarray(
        [
            descriptors.get_descriptor("bond_lengths", atom)[0]
            for atom in base_atoms
        ]
    )
    dihedrals = np.asarray(
        [
            descriptors.get_descriptor("dihedral", atom)[0]
            for atom in base_atoms
        ]
    )
    test_bond_lengths = [
        descriptors.get_descriptor("bond_lengths", atom)[0]
        for atom in base_test_atoms
    ]
    test_dihedrals = [
        descriptors.get_descriptor("dihedral", atom)[0]
        for atom in base_test_atoms
    ]
    sampled_global_indices = train_indices[sampled_indices]

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.scatter(
        bond_lengths[train_indices],
        dihedrals[train_indices],
        color="blue",
        marker="o",
        s=10,
        alpha=0.3,
        label="Train pool",
    )
    ax.scatter(
        test_bond_lengths,
        test_dihedrals,
        color="red",
        marker="o",
        s=10,
        alpha=0.3,
        label="Off-grid test set",
    )
    ax.scatter(
        bond_lengths[valid_indices],
        dihedrals[valid_indices],
        color="green",
        marker="o",
        s=10,
        alpha=0.3,
        label="Validation set",
    )
    ax.scatter(
        bond_lengths[sampled_global_indices],
        dihedrals[sampled_global_indices],
        color="black",
        marker="o",
        s=10,
        alpha=1.0,
        label="Selected samples",
    )
    ax.set(
        title="Selection of transfer-learning samples",
        xlabel="Bond length",
        ylabel="Dihedral angle",
    )
    ax.legend()
    fig.tight_layout()
    filename = "sample_selection.png"
    fig.savefig(run_dir / filename, dpi=150)
    plt.close(fig)
    return filename


def _pca_coordinates(
    descriptor_matrix: np.ndarray, dimensions: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized PCA coordinates, padding unavailable axes with zero."""
    scaled = StandardScaler().fit_transform(descriptor_matrix)
    available = min(dimensions, scaled.shape[0], scaled.shape[1])
    if available < 1:
        raise ValueError("PCA plots require a non-empty descriptor matrix")
    pca = PCA(n_components=available)
    available_coordinates = pca.fit_transform(scaled)
    coordinates = np.zeros((scaled.shape[0], dimensions), dtype=float)
    variance_ratios = np.zeros(dimensions, dtype=float)
    coordinates[:, :available] = available_coordinates
    variance_ratios[:available] = np.nan_to_num(
        pca.explained_variance_ratio_, nan=0.0
    )
    return coordinates, variance_ratios


def _save_pca_selection_plots(
    run_dir: Path,
    descriptor_matrix: np.ndarray,
    sampled_indices: np.ndarray,
    descriptor_name: str,
    selector_name: str,
) -> dict[str, Any]:
    """Save the notebook's 2D PCA plot and three views of its 3D PCA plot."""
    coordinates_2d, variance_2d = _pca_coordinates(descriptor_matrix, 2)
    fig, ax = plt.subplots()
    ax.scatter(
        coordinates_2d[:, 0],
        coordinates_2d[:, 1],
        color="blue",
        marker="o",
        alpha=0.3,
        s=10,
        label="Train pool",
    )
    ax.scatter(
        coordinates_2d[sampled_indices, 0],
        coordinates_2d[sampled_indices, 1],
        color="red",
        marker="o",
        alpha=1.0,
        s=10,
        label="Selected samples",
    )
    ax.set(
        title=f"2D PCA of {descriptor_name} selected by {selector_name}",
        xlabel=f"Principal Component 1 ({variance_2d[0] * 100:.1f}%)",
        ylabel=f"Principal Component 2 ({variance_2d[1] * 100:.1f}%)",
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    pca_2d_filename = "selection_pca_2d.png"
    fig.savefig(run_dir / pca_2d_filename, dpi=150)
    plt.close(fig)

    coordinates_3d, variance_3d = _pca_coordinates(descriptor_matrix, 3)
    views = (
        ("front", 20, 45),
        ("side", 20, 135),
        ("top", 75, 45),
    )
    pca_3d_filenames = []
    for view_name, elevation, azimuth in views:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection="3d")
        ax.set_box_aspect(None, zoom=0.85)
        ax.scatter(
            coordinates_3d[:, 0],
            coordinates_3d[:, 1],
            coordinates_3d[:, 2],
            color="blue",
            marker="o",
            alpha=0.3,
            s=10,
            label="Train pool",
        )
        ax.scatter(
            coordinates_3d[sampled_indices, 0],
            coordinates_3d[sampled_indices, 1],
            coordinates_3d[sampled_indices, 2],
            color="red",
            marker="o",
            alpha=1.0,
            s=10,
            label="Selected samples",
        )
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set(
            title=(
                f"3D PCA of {descriptor_name} selected by {selector_name}\n"
                f"view: elevation {elevation}°, azimuth {azimuth}°"
            ),
            xlabel=f"PC 1 ({variance_3d[0] * 100:.1f}%)",
            ylabel=f"PC 2 ({variance_3d[1] * 100:.1f}%)",
            zlabel=f"PC 3 ({variance_3d[2] * 100:.1f}%)",
        )
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        filename = f"selection_pca_3d_{view_name}.png"
        fig.savefig(run_dir / filename, dpi=150)
        plt.close(fig)
        pca_3d_filenames.append(
            {
                "file": filename,
                "elevation_degrees": elevation,
                "azimuth_degrees": azimuth,
            }
        )

    return {
        "pca_2d_plot": pca_2d_filename,
        "pca_3d_plots": pca_3d_filenames,
    }


def _save_mae_plot(
    run_dir: Path,
    base_mae: float,
    full_mae: float,
    transfer_mae: float,
) -> str:
    categories = ["Base model", "Full HF model", "Transfer model"]
    values = [base_mae, full_mae, transfer_mae]
    fig, ax = plt.subplots()
    bars = ax.bar(categories, values, color=["#377eb8", "#ff7f00", "#fdbf6f"])
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set(ylabel="Mean energy MAE, eV", title="Multi-fidelity transfer learning")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    filename = "energy_mae_comparison.png"
    fig.savefig(run_dir / filename, dpi=150)
    plt.close(fig)
    return filename


def run_config(config_path: Path, output_dir: Path) -> Path:
    """Execute one input configuration and return its result JSON path."""
    (
        modules,
        AtomDataLoaderBuilder,
        Tester,
        extract_latent_space,
        NaiveStrategy,
        Trainer,
        descriptors,
        selectors,
        training,
    ) = _import_project_modules()

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The top-level JSON value must be an object")

    run_dir = _next_run_dir(output_dir)
    result_path = run_dir / "result.json"
    result: dict[str, Any] = {
        "status": "running",
        "input_file": str(config_path.resolve()),
        "run_directory": str(run_dir.resolve()),
    }
    _write_json(result_path, result)

    try:
        seed = int(config.get("seed", SEED))
        max_epochs = int(config.get("max_epochs", MAX_EPOCHS))
        r_max = float(config.get("r_max", R_MAX))
        batch_size = int(config.get("batch_size", BATCH_SIZE))
        transfer_lr = float(config.get("transfer_lr", TRANSFER_LR))
        validation_fraction = float(
            config.get("validation_fraction", VALIDATION_FRACTION)
        )
        patience = int(config.get("patience", PATIENCE))
        device = _validate_device(str(config.get("device", DEVICE)))

        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("'validation_fraction' must be between 0 and 1")
        if max_epochs < 1 or batch_size < 1:
            raise ValueError("'max_epochs' and 'batch_size' must be positive")

        training.seed_everything(seed)

        base_xyz = _path(_required(config, "base_xyz"), config_path)
        transfer_xyz = _path(_required(config, "transfer_xyz"), config_path)
        base_test_xyz = _path(_required(config, "base_test_xyz"), config_path)
        transfer_test_xyz = _path(
            _required(config, "transfer_test_xyz"), config_path
        )
        base_model_path = _path(
            _required(config, "base_model_path"), config_path
        )
        full_model_path = _path(
            _required(config, "full_model_path"), config_path
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
        train_indices, valid_indices = train_test_split(
            all_indices,
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
        )
        base_train_atoms = [base_atoms[i] for i in train_indices]
        transfer_train_atoms = [transfer_atoms[i] for i in train_indices]
        transfer_valid_atoms = [transfer_atoms[i] for i in valid_indices]

        split_plot = _save_split_plot(
            run_dir,
            base_atoms,
            base_test_atoms,
            train_indices,
            valid_indices,
            descriptors,
        )

        data_builder = AtomDataLoaderBuilder(
            cutoff=r_max,
            energy_key=str(config.get("energy_key", "REF_energy")),
            forces_key=str(config.get("forces_key", "REF_forces")),
        )
        trainer = Trainer(
            max_epochs=max_epochs,
            early_stopping=bool(config.get("early_stopping", True)),
            patience=patience,
            restore_best=bool(config.get("restore_best", True)),
            device=device,
            verbose=bool(config.get("verbose", True)),
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

        base_test_loader = data_builder.load(
            base_test_atoms, batch_size=batch_size, shuffle=False
        )
        transfer_test_loader = data_builder.load(
            transfer_test_atoms, batch_size=batch_size, shuffle=False
        )
        base_model = _load_model(base_model_path, device)
        full_model = _load_model(full_model_path, device)
        base_metrics = _evaluate(base_model, base_test_loader, tester)
        full_metrics = _evaluate(full_model, transfer_test_loader, tester)

        descriptor_name = str(_required(config, "descriptor"))
        descriptor_kwargs = dict(config.get("descriptor_kwargs", {}))
        if descriptor_name == "latent_space":
            unshuffled_loader = data_builder.load(
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
        training.seed_everything(seed)
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

        transfer_train_loader = data_builder.load(
            sampled_atoms, batch_size=batch_size, shuffle=True
        )
        transfer_valid_loader = data_builder.load(
            transfer_valid_atoms, batch_size=batch_size, shuffle=False
        )
        transfer_model = NaiveStrategy().apply(base_model)
        transfer_optimizer = torch.optim.Adam(
            transfer_model.parameters(), lr=transfer_lr
        )
        training.seed_everything(seed)
        started_at = time.time()
        transfer_model, transfer_history = trainer.train_model(
            transfer_model,
            transfer_train_loader,
            transfer_valid_loader,
            transfer_optimizer,
            loss_fn,
        )
        training_seconds = time.time() - started_at
        transfer_metrics = _evaluate(
            transfer_model, transfer_test_loader, tester
        )
        transfer_metrics["best_epoch"] = int(transfer_history["best_epoch"])

        loss_plot = _save_loss_plot(run_dir, transfer_history)
        mae_plot = _save_mae_plot(
            run_dir,
            base_metrics["energy_mae_ev"],
            full_metrics["energy_mae_ev"],
            transfer_metrics["energy_mae_ev"],
        )
        result.update(
            {
                "status": "completed",
                "config": config,
                "seed": seed,
                "device": str(device),
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
                    "best_epoch": transfer_history["best_epoch"],
                    "training_seconds": training_seconds,
                    "history": transfer_history,
                },
                "artifacts": {
                    "data_split_plot": split_plot,
                    "sample_selection_plot": selection_plot,
                    **pca_plots,
                    "transfer_loss_plot": loss_plot,
                    "energy_mae_comparison_plot": mae_plot,
                },
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
    config_paths = sorted(input_dir.glob("*.json"))
    if not config_paths:
        print(f"No JSON input files found in {input_dir}", file=sys.stderr)
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
