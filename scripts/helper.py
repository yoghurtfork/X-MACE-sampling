"""Shared utilities for the X-MACE training and transfer scripts."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import ase.io
import matplotlib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

SEED = 42
MAX_EPOCHS = 100
R_MAX = 5.0
BATCH_SIZE = 64
TRANSFER_LR = 5.0e-4
SCRATCH_LR = 1.0e-3
DEVICE = "cpu"
VALIDATION_FRACTION = 0.1
BASE_E0S = {"C": -1032.083979117871, "H": -15.357929595328724}
FULL_E0S = {"C": -1035.5115207879423, "H": -15.712048126191444}
STATE_LABELS = ("S0", "S1", "S2")


def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)


def _import_project_modules() -> tuple[Any, ...]:
    """Import local and X-MACE modules after making the project importable."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from mace import modules
    from mace.data.atom_data_loader import AtomDataLoaderBuilder
    from mace.testing import Tester, extract_latent_space
    from mace.training import (
        NaiveStrategy,
        Trainer,
        initialise_autoencoder,
    )

    import sampling_methods.descriptors as descriptors
    import sampling_methods.selectors as selectors
    return (
        modules,
        AtomDataLoaderBuilder,
        Tester,
        extract_latent_space,
        NaiveStrategy,
        Trainer,
        initialise_autoencoder,
        descriptors,
        selectors,
    )


def _required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required configuration key: {key!r}")
    return config[key]


def _e0s_from_config(
    config: dict[str, Any], key: str, defaults: dict[str, float]
) -> dict[str, float]:
    """Return defaults overlaid with an optional JSON E0 mapping."""
    overrides = config.get(key, {})
    if not isinstance(overrides, dict):
        raise ValueError(f"'{key}' must be a JSON object")

    e0s = defaults.copy()
    for element, energy in overrides.items():
        if not isinstance(element, str) or not element:
            raise ValueError(f"'{key}' element names must be non-empty strings")
        if isinstance(energy, bool) or not isinstance(energy, (int, float)):
            raise ValueError(f"'{key}' values must be JSON numbers")
        e0s[element] = float(energy)
    return e0s


def _trainer_options_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract validated X-MACE Trainer options from a run configuration."""
    options: dict[str, Any] = {}
    for key, default in (
        ("early_stopping", True),
        ("restore_best", True),
        ("verbose", True),
    ):
        value = config.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"'{key}' must be a JSON boolean")
        options[key] = value

    for key, minimum, maximum, allow_minimum in (
        ("optimiser_lr", 0.0, None, False),
        ("optimiser_weight_decay", 0.0, None, True),
        ("max_grad_norm", 0.0, None, False),
        ("scheduler_lr_factor", 0.0, 1.0, False),
        ("ema_decay", 0.0, 1.0, True),
    ):
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'{key}' must be a JSON number")
        value = float(value)
        below_minimum = value < minimum or (
            value == minimum and not allow_minimum
        )
        if below_minimum or (maximum is not None and value > maximum):
            if maximum is None:
                if allow_minimum:
                    raise ValueError(f"'{key}' must be non-negative")
                raise ValueError(f"'{key}' must be positive")
            lower_bound = "at least 0" if allow_minimum else "greater than 0"
            raise ValueError(
                f"'{key}' must be {lower_bound} and at most {maximum}"
            )
        options[key] = value

    if "scheduler_patience" in config:
        value = config["scheduler_patience"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("'scheduler_patience' must be a positive JSON integer")
        options["scheduler_patience"] = value

    patience_key = (
        "stopping_patience"
        if "stopping_patience" in config
        else "patience" if "patience" in config else None
    )
    if patience_key is not None:
        value = config[patience_key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"'{patience_key}' must be a positive JSON integer")
        options["stopping_patience"] = value
    return options


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
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, default=_json_value, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _validate_device(name: str) -> torch.device:
    if name not in {"cpu", "cuda"}:
        raise ValueError("'device' must be either 'cpu' or 'cuda'")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _load_model(path: Path, device: torch.device) -> torch.nn.Module:
    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    model = torch.load(path, map_location=device, weights_only=False)
    return model.to(device)


def _evaluate(
    model: torch.nn.Module, loader: Any, tester: Any
) -> dict[str, Any]:
    model.eval()
    tester.run_test(model, loader)
    return {
        "energy_mae_ev": float(tester.get_energy_mae()),
        "force_mae_ev_per_ang": float(tester.get_force_mae()),
        "energy_mae_by_state_ev": _maes_by_state(
            tester.get_energy_mae_by_state(), "energy"
        ),
        "force_mae_by_state_ev_per_ang": _maes_by_state(
            tester.get_force_mae_by_state(), "force"
        ),
    }


def _maes_by_state(values: Any, metric_name: str) -> dict[str, float]:
    """Label the three X-MACE per-state MAEs as S0, S1, and S2."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    if isinstance(values, dict):
        values = list(values.values())
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(STATE_LABELS):
        raise ValueError(
            f"Tester returned {len(values)} {metric_name} MAEs; expected "
            f"{len(STATE_LABELS)} for S0, S1, and S2"
        )
    return {
        state: float(value) for state, value in zip(STATE_LABELS, values)
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


def _save_loss_plot(
    run_dir: Path,
    history: dict[str, Any],
    *,
    title: str = "Transfer learning",
    filename: str = "transfer_loss.png",
) -> str:
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
        title=title,
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=150)
    plt.close(fig)
    return filename


def _save_epoch_mae_plot(
    run_dir: Path,
    history: dict[str, Any],
    *,
    title: str,
    filename: str,
) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    axes[0].plot(
        history["epoch"],
        history["valid_energy_mae"],
        marker="o",
        linewidth=1.8,
        color="#377eb8",
    )
    axes[0].set_ylabel("Energy MAE, eV")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(
        history["epoch"],
        history["valid_force_mae"],
        marker="s",
        linewidth=1.8,
        color="#ff7f00",
    )
    axes[1].set(xlabel="Epoch", ylabel="Force MAE, eV/Å")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
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
    base_metrics: dict[str, Any],
    full_metrics: dict[str, Any],
    transfer_metrics: dict[str, Any],
    transfer_best_epoch: Any,
    *,
    cross_validation: bool,
) -> str:
    categories = ["Base model", "Full HF model", "Transfer model"]
    state_colors = ["#377eb8", "#ff7f00", "#4daf4a"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    if cross_validation:
        best_epoch_value, best_epoch_variance = transfer_best_epoch
        best_epoch_error = np.sqrt(best_epoch_variance)
    else:
        best_epoch_value = float(transfer_best_epoch)
        best_epoch_error = 0.0
    best_bars = axes[0].bar(
        ["Transfer model"],
        [best_epoch_value],
        yerr=[best_epoch_error] if cross_validation else None,
        capsize=5 if cross_validation else 0,
        color="#fdbf6f",
    )
    axes[0].bar_label(best_bars, fmt="%.1f", padding=3)
    axes[0].set(ylabel="Best epoch", title="Best training epoch")
    axes[0].grid(axis="y", alpha=0.3)

    for ax, (metric, ylabel, title) in zip(
        axes[1:],
        (
            (
                "energy_mae_by_state_ev",
                "Energy MAE, eV",
                "Final energy MAE by state",
            ),
            (
                "force_mae_by_state_ev_per_ang",
                "Force MAE, eV/Å",
                "Final force MAE by state",
            ),
        ),
    ):
        _plot_state_mae_bars(
            ax,
            categories,
            (base_metrics, full_metrics, transfer_metrics),
            metric,
            cross_validation=cross_validation,
            colors=state_colors,
        )
        ax.set(ylabel=ylabel, title=title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Multi-fidelity transfer learning")
    fig.tight_layout()
    filename = "final_metrics_comparison.png"
    fig.savefig(run_dir / filename, dpi=150)
    plt.close(fig)
    return filename


def _plot_state_mae_bars(
    ax: Any,
    categories: list[str],
    metrics_by_model: tuple[dict[str, Any], ...],
    metric: str,
    *,
    cross_validation: bool,
    colors: list[str],
) -> None:
    """Draw grouped S0/S1/S2 MAE bars for the supplied models."""
    positions = np.arange(len(categories), dtype=float)
    width = 0.24
    offsets = (np.arange(len(STATE_LABELS)) - 1) * width
    for offset, state, color in zip(offsets, STATE_LABELS, colors):
        values = []
        errors = []
        for model_metrics in metrics_by_model:
            value = model_metrics[metric][state]
            if cross_validation and isinstance(value, dict):
                values.append(value["mean"])
                errors.append(np.sqrt(value["variance"]))
            else:
                values.append(value)
                errors.append(0.0)
        bars = ax.bar(
            positions + offset,
            values,
            width,
            yerr=errors if cross_validation else None,
            capsize=4 if cross_validation else 0,
            color=color,
            label=state,
        )
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
    ax.set_xticks(positions, categories)
    ax.legend(title="State", fontsize=8)


def _aggregate_fold_metrics(
    fold_results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    aggregate = {}
    for metric in ("energy_mae_ev", "force_mae_ev_per_ang"):
        values = np.asarray(
            [fold["metrics"][metric] for fold in fold_results.values()],
            dtype=float,
        )
        aggregate[metric] = {
            "mean": float(np.mean(values)),
            "variance": float(np.var(values)),
        }
    for metric in (
        "energy_mae_by_state_ev",
        "force_mae_by_state_ev_per_ang",
    ):
        aggregate[metric] = {}
        for state in STATE_LABELS:
            values = np.asarray(
                [
                    fold["metrics"][metric][state]
                    for fold in fold_results.values()
                ],
                dtype=float,
            )
            aggregate[metric][state] = {
                "mean": float(np.mean(values)),
                "variance": float(np.var(values)),
            }
    return aggregate


def _train_k_fold_models(
    *,
    initial_model: torch.nn.Module,
    all_atoms: list[Any],
    test_atoms: list[Any],
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
    e0s: dict[str, float],
    on_fold_complete: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train, test, plot, and save a set of X-MACE K-fold models."""
    if not 2 <= k <= len(all_atoms):
        raise ValueError(
            f"'k' must be between 2 and the {model_prefix} dataset size "
            f"({len(all_atoms)})"
        )
    builder = data_builder_class(
        cutoff=r_max,
        energy_key=energy_key,
        forces_key=forces_key,
        E0s=e0s,
    )
    test_loader = builder.load(
        test_atoms, batch_size=batch_size, shuffle=False
    )
    seed_everything(seed)
    started_at = time.time()
    fold_results: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, str]] = {}
    model_paths: dict[str, str] = {}
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    for fold_number, (train_indices, valid_indices) in enumerate(
        splitter.split(range(len(all_atoms))), start=1
    ):
        train_loader = builder.load(
            [all_atoms[index] for index in train_indices],
            batch_size=batch_size,
            shuffle=True,
        )
        valid_loader = builder.load(
            [all_atoms[index] for index in valid_indices],
            batch_size=batch_size,
            shuffle=False,
        )
        fold_model = deepcopy(initial_model).to(device)
        optimizer = torch.optim.Adam(
            fold_model.parameters(), lr=learning_rate
        )
        trainer = trainer_class(
            max_epochs=max_epochs,
            device=device,
            **trainer_options,
        )
        fold_model, history = trainer.train_model(
            fold_model, train_loader, valid_loader, optimizer, loss_fn
        )
        metrics = _evaluate(fold_model, test_loader, tester)
        metrics["best_epoch"] = int(history["best_epoch"])
        fold_model = fold_model.cpu()
        model_key = f"model_{fold_number}"
        model_path = (
            run_dir / f"{model_prefix}_fold_{fold_number}.pt"
        ).resolve()
        torch.save(fold_model, model_path)
        loss_plot = _save_loss_plot(
            run_dir,
            history,
            title=f"{model_prefix.replace('_', ' ').title()} fold {fold_number}",
            filename=f"{model_prefix}_fold_{fold_number}_loss.png",
        )
        mae_plot = _save_epoch_mae_plot(
            run_dir,
            history,
            title=(
                f"{model_prefix.replace('_', ' ').title()} "
                f"fold {fold_number} validation MAE"
            ),
            filename=f"{model_prefix}_fold_{fold_number}_validation_mae.png",
        )
        fold_results[model_key] = {
            "history": history,
            "metrics": metrics,
            "model_path": str(model_path),
        }
        model_paths[model_key] = str(model_path)
        artifacts[model_key] = {
            "loss_plot": loss_plot,
            "validation_mae_plot": mae_plot,
        }
        snapshot = _cross_validation_snapshot(
            fold_results,
            artifacts,
            model_paths,
            total_folds=k,
            started_at=started_at,
        )
        if on_fold_complete is not None:
            on_fold_complete(snapshot)

    return _cross_validation_snapshot(
        fold_results,
        artifacts,
        model_paths,
        total_folds=k,
        started_at=started_at,
    )


def _cross_validation_snapshot(
    fold_results: dict[str, dict[str, Any]],
    artifacts: dict[str, dict[str, str]],
    model_paths: dict[str, str],
    *,
    total_folds: int,
    started_at: float,
) -> dict[str, Any]:
    best_epochs = np.asarray(
        [fold["metrics"]["best_epoch"] for fold in fold_results.values()],
        dtype=float,
    )
    return {
        "folds": fold_results,
        "combined_validation": {
            "best_epoch": [
                float(np.mean(best_epochs)),
                float(np.var(best_epochs)),
            ]
        },
        "aggregate_test_metrics": _aggregate_fold_metrics(fold_results),
        "training_seconds": time.time() - started_at,
        "model_paths": model_paths,
        "artifacts": artifacts,
        "completed_folds": len(fold_results),
        "total_folds": total_folds,
    }


def _save_fold_selection_plot(
    *,
    run_dir: Path,
    base_atoms: list[Any],
    base_test_atoms: list[Any],
    sampled_global_indices: np.ndarray,
    fold_train_indices: np.ndarray,
    fold_valid_indices: np.ndarray,
    fold_number: int,
    model_prefix: str,
    descriptors: Any,
) -> str:
    """Plot one transfer fold's train/validation membership."""
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
    test_bonds = [
        descriptors.get_descriptor("bond_lengths", atom)[0]
        for atom in base_test_atoms
    ]
    test_dihedrals = [
        descriptors.get_descriptor("dihedral", atom)[0]
        for atom in base_test_atoms
    ]
    fold_train_global = sampled_global_indices[fold_train_indices]
    fold_valid_global = sampled_global_indices[fold_valid_indices]

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.scatter(
        bond_lengths,
        dihedrals,
        color="blue",
        marker="o",
        s=10,
        alpha=0.25,
        label="Candidate pool",
    )
    ax.scatter(
        test_bonds,
        test_dihedrals,
        color="red",
        marker="o",
        s=10,
        alpha=0.3,
        label="Off-grid test set",
    )
    ax.scatter(
        bond_lengths[fold_train_global],
        dihedrals[fold_train_global],
        color="black",
        marker="o",
        s=12,
        label="Fold training samples",
    )
    ax.scatter(
        bond_lengths[fold_valid_global],
        dihedrals[fold_valid_global],
        color="green",
        marker="o",
        s=16,
        label="Fold validation samples",
    )
    ax.set(
        title=(
            f"{model_prefix.replace('_', ' ').title()} "
            f"selection: fold {fold_number}"
        ),
        xlabel="Bond length",
        ylabel="Dihedral angle",
    )
    ax.legend()
    fig.tight_layout()
    filename = f"{model_prefix}_fold_{fold_number}_selection.png"
    fig.savefig(run_dir / filename, dpi=150)
    plt.close(fig)
    return filename


def _save_scratch_mae_plot(
    run_dir: Path,
    base_metrics: dict[str, Any],
    full_metrics: dict[str, Any],
    base_best_epoch: Any,
    full_best_epoch: Any,
    *,
    cross_validation: bool,
) -> str:
    categories = ["Base model", "Full HF model"]
    state_colors = ["#377eb8", "#ff7f00", "#4daf4a"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    if cross_validation:
        best_values = [base_best_epoch[0], full_best_epoch[0]]
        best_errors = [
            np.sqrt(base_best_epoch[1]),
            np.sqrt(full_best_epoch[1]),
        ]
    else:
        best_values = [base_best_epoch, full_best_epoch]
        best_errors = None
    best_bars = axes[0].bar(
        categories,
        best_values,
        yerr=best_errors,
        capsize=5 if cross_validation else 0,
        color=["#377eb8", "#ff7f00"],
    )
    axes[0].bar_label(best_bars, fmt="%.1f", padding=3)
    axes[0].set(ylabel="Best epoch", title="Best training epoch")
    axes[0].grid(axis="y", alpha=0.3)

    for ax, (metric, ylabel, title) in zip(
        axes[1:],
        (
            (
                "energy_mae_by_state_ev",
                "Energy MAE, eV",
                "Final energy MAE by state",
            ),
            (
                "force_mae_by_state_ev_per_ang",
                "Force MAE, eV/Å",
                "Final force MAE by state",
            ),
        ),
    ):
        _plot_state_mae_bars(
            ax,
            categories,
            (base_metrics, full_metrics),
            metric,
            cross_validation=cross_validation,
            colors=state_colors,
        )
        ax.set(ylabel=ylabel, title=title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Models trained from scratch")
    fig.tight_layout()
    filename = "final_metrics_comparison.png"
    fig.savefig(run_dir / filename, dpi=150)
    plt.close(fig)
    return filename


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
    trainer_options: dict[str, Any],
    energy_key: str,
    forces_key: str,
    preset: str,
    load_base: str | None,
    e0s: dict[str, float],
) -> dict[str, Any]:
    builder = builder_class(
        cutoff=r_max,
        energy_key=energy_key,
        forces_key=forces_key,
        E0s=e0s,
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
    seed_everything(seed)
    model = initialise_autoencoder(
        builder.get_metadata(), preset=preset, load_base=load_base
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = trainer_class(
        max_epochs=max_epochs,
        device=device,
        **trainer_options,
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


def _is_scratch_config(path: Path) -> bool:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(config, dict) and config.get("transfer_learning") is False
