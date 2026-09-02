"""Atomic run-state persistence and plot generation for training runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def reserve_run_dir(output_dir: Path) -> Path:
    """Atomically reserve the next available ``run_<index>`` directory."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        candidate = output_dir / f"run_{index}"
        try:
            candidate.mkdir()
        except FileExistsError:
            index += 1
            continue
        return candidate


def save_model(model: Any, path: Path) -> Path:
    """Atomically save a torch model without moving it off its active device."""
    import torch

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(model, temporary_path)
    temporary_path.replace(path)
    return path


class RunState:
    """The incrementally persisted result document for one reserved run."""

    def __init__(
        self,
        run_dir: Path,
        input_file: Path,
        config: dict[str, Any],
        warnings: list[str],
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"Reserved run directory does not exist: {self.run_dir}")
        self.result_path = self.run_dir / "result.json"
        self.result: dict[str, Any] = {
            "status": "running",
            "input_file": str(Path(input_file).resolve()),
            "run_directory": str(self.run_dir),
            "config": config,
            "warnings": list(warnings),
            "stages": {},
        }
        self.write()

    def write(self) -> Path:
        """Atomically persist the current result document."""
        write_json(self.result_path, self.result)
        return self.result_path

    def update_stage(self, stage: str, updates: Mapping[str, Any]) -> None:
        """Merge stage runtime information and persist it immediately."""
        if stage not in {"lf", "hf", "transfer"}:
            raise ValueError("'stage' must be one of: lf, hf, transfer")
        current = self.result["stages"].setdefault(stage, {"status": "running"})
        current.update(dict(updates))
        self.write()

    def add_warning(self, warning: str) -> None:
        """Record a runtime warning and persist it alongside config warnings."""
        self.result["warnings"].append(warning)
        self.write()

    def complete(self) -> Path:
        """Mark the whole run completed after all requested stages finish."""
        self.result["status"] = "completed"
        return self.write()

    def interrupt(self, details: str | None = None) -> Path:
        """Mark the run interrupted without losing completed-stage data."""
        self.result["status"] = "interrupted"
        if details:
            self.result["interruption"] = {"message": details}
        return self.write()

    def fail(self, error: BaseException | str) -> Path:
        """Mark the run failed and retain a concise, JSON-safe error record."""
        self.result["status"] = "failed"
        self.result["error"] = {
            "type": type(error).__name__ if isinstance(error, BaseException) else "Error",
            "message": str(error),
        }
        return self.write()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write JSON with scalar arrays compacted to one line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_value = json_safe(value)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(_format_json(safe_value) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def json_safe(value: Any) -> Any:
    """Convert supported scientific values to strict, finite JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN and infinity cannot be serialized to result JSON")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if _is_torch_device(value):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Result JSON object keys must be strings")
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def save_loss_plot(run_dir: Path, history: Mapping[str, Any], *, title: str, filename: str) -> str:
    """Save the training/validation loss figure for one stage or fold."""
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(4, 2.5))
    axis.plot(history["epoch"], history["train_loss"], marker="o", linewidth=1.8, label="Train loss", alpha=0.8)
    axis.plot(history["epoch"], history["valid_loss"], marker="s", linewidth=1.8, label="Validation loss", alpha=0.8)
    axis.set(xlabel="Epoch", ylabel="Loss", title=title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    return _save_figure(fig, Path(run_dir), filename)


def save_epoch_mae_plot(run_dir: Path, history: Mapping[str, Any], *, title: str, filename: str) -> str:
    """Save the validation energy/force MAE-over-epochs figure."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    axes[0].plot(history["epoch"], history["valid_energy_mae"], marker="o", linewidth=1.8, color="#377eb8")
    axes[0].set_ylabel("Energy MAE, eV")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["epoch"], history["valid_force_mae"], marker="s", linewidth=1.8, color="#ff7f00")
    axes[1].set(xlabel="Epoch", ylabel="Force MAE, eV/Å")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return _save_figure(fig, Path(run_dir), filename)


def save_selection_plot(
    run_dir: Path,
    coordinates: np.ndarray,
    selected_indices: np.ndarray,
    *,
    filename: str,
    title: str,
    candidate_label: str = "Candidate pool",
) -> str:
    """Save a two-dimensional candidate/selected-sample scatter plot."""
    coordinates = np.asarray(coordinates, dtype=float)
    selected_indices = np.asarray(selected_indices, dtype=int)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("Selection plots require coordinates with at least two columns")
    if np.any(selected_indices < 0) or np.any(selected_indices >= len(coordinates)):
        raise ValueError("Selection plot indices are out of range")
    plt = _pyplot()
    fig, axis = plt.subplots()
    axis.scatter(coordinates[:, 0], coordinates[:, 1], color="tab:blue", s=12, alpha=0.3, label=candidate_label)
    axis.scatter(coordinates[selected_indices, 0], coordinates[selected_indices, 1], color="black", s=16, label="Selected samples")
    axis.set(title=title, xlabel="Coordinate 1", ylabel="Coordinate 2")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    return _save_figure(fig, Path(run_dir), filename)


def save_pca_selection_plot(
    run_dir: Path,
    descriptor_matrix: np.ndarray,
    selected_indices: np.ndarray,
    *,
    filename: str,
    title: str,
) -> str:
    """Standardize descriptors, project them into 2D PCA, then plot selection."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    matrix = np.asarray(descriptor_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("PCA selection plots require a non-empty descriptor matrix")
    scaled = StandardScaler().fit_transform(matrix)
    dimensions = min(2, scaled.shape[0], scaled.shape[1])
    coordinates = PCA(n_components=dimensions).fit_transform(scaled)
    if dimensions == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(coordinates))))
    return save_selection_plot(run_dir, coordinates, selected_indices, filename=filename, title=title, candidate_label="Descriptor pool")


def _is_torch_device(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.device)


def _is_scalar_array(value: list[Any]) -> bool:
    return all(item is None or isinstance(item, (str, bool, int, float)) for item in value)


def _format_json(value: Any, level: int = 0) -> str:
    """Pretty-print objects while retaining scalar arrays on a single line."""
    indent = "  " * level
    child_indent = "  " * (level + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        entries = [
            f"{child_indent}{json.dumps(key)}: {_format_json(item, level + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(entries) + "\n" + indent + "}"
    if isinstance(value, list):
        if _is_scalar_array(value):
            return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(", ", ": "))
        if not value:
            return "[]"
        return "[\n" + ",\n".join(
            f"{child_indent}{_format_json(item, level + 1)}" for item in value
        ) + "\n" + indent + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(figure: Any, run_dir: Path, filename: str) -> str:
    path = run_dir / filename
    figure.savefig(path, dpi=150)
    _pyplot().close(figure)
    return str(path.resolve())
