"""Atomic, resumable result-state storage for active-learning runs."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from scripts.helper import _write_json


_SCHEMA_VERSION = 1
_RESUMABLE_STATUSES = {"running", "interrupted"}


@dataclass(frozen=True)
class StateStore:
    """The atomic ``result.json`` store for one active-learning run."""

    result_path: Path

    def save(self, state: dict[str, Any]) -> None:
        """Atomically write state after updating its timestamp."""
        state["updated_at"] = _timestamp()
        _write_json(self.result_path, state)

    def load(self) -> dict[str, Any]:
        """Read a previously written state document."""
        import json

        if not self.result_path.is_file():
            raise FileNotFoundError(f"Result file does not exist: {self.result_path}")
        value = json.loads(self.result_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Result JSON must contain an object")
        if value.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Result JSON has an unsupported schema version")
        return value

    def resume(self, identity: Mapping[str, Any]) -> dict[str, Any]:
        """Load only an unfinished run with an identical immutable identity."""
        state = self.load()
        if state.get("status") not in _RESUMABLE_STATUSES:
            raise ValueError(
                f"Cannot resume a run with status {state.get('status')!r}"
            )
        if state.get("resume_identity") != dict(identity):
            raise ValueError("Result JSON does not match this active-learning input")
        return state


def make_resume_identity(
    *,
    config_path: Path,
    lf_checkpoint: Path,
    hf_xyz: Path,
    grid_shape: tuple[int, int],
) -> dict[str, Any]:
    """Build the immutable identity checked before resuming a run."""
    return {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "lf_checkpoint": str(lf_checkpoint.resolve()),
        "hf_xyz": str(hf_xyz.resolve()),
        "grid_shape": list(grid_shape),
    }


def new_state(
    *,
    input_file: Path,
    run_directory: Path,
    config: Mapping[str, Any],
    resume_identity: Mapping[str, Any],
    grid_size: int,
    grid_shape: tuple[int, int],
    energy_key: str,
    forces_key: str,
    initial_acquired_indices: Iterable[int],
) -> dict[str, Any]:
    """Create the initial in-memory result document for a new run."""
    initial = _validated_indices(initial_acquired_indices, grid_size)
    if not initial:
        raise ValueError("At least one initial HF geometry must be acquired")
    now = _timestamp()
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": "running",
        "input_file": str(input_file.resolve()),
        "run_directory": str(run_directory.resolve()),
        "config": deepcopy(dict(config)),
        "resume_identity": deepcopy(dict(resume_identity)),
        "hf_grid": {
            "size": grid_size,
            "grid_shape": list(grid_shape),
            "energy_key": energy_key,
            "forces_key": forces_key,
        },
        "initial_acquired_indices": initial,
        "acquired_indices": initial.copy(),
        "rounds": [],
        "final_production_model": {"status": "pending"},
        "created_at": now,
        "updated_at": now,
    }


def start_round(
    state: dict[str, Any], *, round_number: int, acquired_before: Iterable[int]
) -> dict[str, Any]:
    """Append and return an in-progress round record."""
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 0:
        raise ValueError("'round_number' must be a non-negative integer")
    if any(round_record["round_number"] == round_number for round_record in state["rounds"]):
        raise ValueError(f"Round {round_number} already exists in result state")
    acquired = _state_indices(state, acquired_before)
    record = {
        "round_number": round_number,
        "status": "training",
        "acquired_before_indices": acquired,
    }
    state["rounds"].append(record)
    return record


def checkpoint_current_round(
    state: dict[str, Any], *, committee: Mapping[str, Any]
) -> None:
    """Store the latest helper checkpoint snapshot for the active round."""
    record = _current_round(state)
    if record["status"] != "training":
        raise ValueError("Only a training round can receive a committee checkpoint")
    record["committee"] = deepcopy(dict(committee))


def complete_current_round(
    state: dict[str, Any],
    *,
    committee: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    selection: Mapping[str, Any],
    acquired_after: Iterable[int],
) -> None:
    """Persist completed committee, acquisition summary, and enlarged pool."""
    record = _current_round(state)
    if record["status"] != "training":
        raise ValueError("Only a training round can be completed")
    acquired = _state_indices(state, acquired_after)
    previous = set(record["acquired_before_indices"])
    if not previous.issubset(acquired):
        raise ValueError("A completed round cannot remove acquired geometries")
    record.update(
        {
            "status": "completed",
            "committee": deepcopy(dict(committee)),
            "uncertainty": deepcopy(dict(uncertainty)),
            "selection": deepcopy(dict(selection)),
        }
    )
    state["acquired_indices"] = acquired


def interrupt_current_round(state: dict[str, Any]) -> None:
    """Mark an in-progress round and the overall run as interrupted."""
    record = _current_round(state)
    if record["status"] != "training":
        raise ValueError("Only a training round can be interrupted")
    record["status"] = "interrupted"
    state["status"] = "interrupted"


def restart_interrupted_round(state: dict[str, Any]) -> dict[str, Any]:
    """Reset the latest interrupted round for deterministic retraining."""
    record = _current_round(state)
    if state.get("status") != "interrupted" or record.get("status") != "interrupted":
        raise ValueError("Only the latest interrupted round can be restarted")
    round_number = record["round_number"]
    record.clear()
    record.update(
        {
            "round_number": round_number,
            "status": "training",
            "acquired_before_indices": list(state["acquired_indices"]),
        }
    )
    state["status"] = "running"
    return record


def begin_final_production_model(state: dict[str, Any]) -> None:
    """Record that validation-free final production training has started."""
    if state["final_production_model"].get("status") not in {"pending", "interrupted"}:
        raise ValueError("Final production-model training has already completed")
    state["final_production_model"] = {"status": "training"}


def complete_final_production_model(
    state: dict[str, Any], *, result: Mapping[str, Any]
) -> None:
    """Record final production artifacts and mark the run complete."""
    if state["final_production_model"].get("status") != "training":
        raise ValueError("Final production-model training has not started")
    state["final_production_model"] = {
        "status": "completed",
        **deepcopy(dict(result)),
    }
    state["status"] = "completed"


def _current_round(state: Mapping[str, Any]) -> dict[str, Any]:
    if not state.get("rounds"):
        raise ValueError("No active-learning round has been started")
    return state["rounds"][-1]


def _state_indices(state: Mapping[str, Any], indices: Iterable[int]) -> list[int]:
    grid = state.get("hf_grid")
    if not isinstance(grid, Mapping) or not isinstance(grid.get("size"), int):
        raise ValueError("Result state has no valid HF-grid size")
    return _validated_indices(indices, grid["size"])


def _validated_indices(indices: Iterable[int], grid_size: int) -> list[int]:
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 1:
        raise ValueError("HF-grid size must be a positive integer")
    resolved = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("Acquired indices must be integers")
        index = int(index)
        if not 0 <= index < grid_size:
            raise ValueError(f"Acquired index {index} is outside [0, {grid_size - 1}]")
        resolved.append(index)
    if len(set(resolved)) != len(resolved):
        raise ValueError("Acquired indices must be unique")
    return sorted(resolved)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
