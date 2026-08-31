"""Dataset, split, and loader helpers for unified LF/HF training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class StageData:
    """Loaded data and its builder for one LF, HF, or transfer stage."""

    atoms: list[Any]
    test_sets: dict[str, list[Any]]
    builder: Any
    resolved_e0s: dict[str, float]


def read_atoms(path: Path, limit: int | None = None) -> list[Any]:
    """Read a training or test XYZ file, optionally limiting its geometries."""
    import ase.io

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"XYZ file does not exist: {path}")
    index = ":" if limit is None else f":{limit}"
    atoms = ase.io.read(path, index=index)
    return atoms if isinstance(atoms, list) else [atoms]


def read_test_sets(paths: Iterable[Path]) -> dict[str, list[Any]]:
    """Read ordered test files into stable ``test_1``, ``test_2`` keys."""
    return {
        f"test_{index}": read_atoms(path)
        for index, path in enumerate(paths, start=1)
    }


def make_builder(config: dict[str, Any], fidelity: str) -> Any:
    """Construct an X-MACE loader builder for LF or HF-labelled data."""
    if fidelity not in {"lf", "hf"}:
        raise ValueError("'fidelity' must be either 'lf' or 'hf'")
    from mace.data.atom_data_loader import AtomDataLoaderBuilder

    return AtomDataLoaderBuilder(
        cutoff=config["r_max"],
        energy_key=config["energy_key"],
        forces_key=config["forces_key"],
        E0s=config.get(f"{fidelity}_E0s"),
    )


def load_stage_data(config: dict[str, Any], fidelity: str) -> StageData:
    """Load one fidelity's training/test data and resolve its fitted E0s."""
    if fidelity not in {"lf", "hf"}:
        raise ValueError("'fidelity' must be either 'lf' or 'hf'")
    atoms = read_atoms(
        config[f"{fidelity}_xyz"], config.get(f"{fidelity}_n_geometries")
    )
    if len(atoms) < 2:
        raise ValueError(f"'{fidelity}_xyz' must contain at least two geometries")
    test_sets = read_test_sets(config[f"{fidelity}_test_xyz"])
    builder = make_builder(config, fidelity)
    # Fitting E0 values from the training pool must happen before loading test
    # data so held-out geometries cannot affect the fitted offsets.
    builder.load(atoms, batch_size=config["batch_size"], shuffle=False)
    return StageData(atoms, test_sets, builder, resolved_e0s(builder))


def make_loader(builder: Any, atoms: list[Any], batch_size: int, *, shuffle: bool) -> Any:
    """Build one dataloader with the common X-MACE calling convention."""
    return builder.load(atoms, batch_size=batch_size, shuffle=shuffle)


def make_test_loaders(
    builder: Any, test_sets: dict[str, list[Any]], batch_size: int
) -> dict[str, Any]:
    """Build non-shuffled loaders for every named test set."""
    return {
        name: make_loader(builder, atoms, batch_size, shuffle=False)
        for name, atoms in test_sets.items()
    }


def fixed_split(size: int, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train and validation indices for a fixed split."""
    _validate_split_size(size)
    from sklearn.model_selection import train_test_split

    return tuple(
        np.asarray(indices, dtype=int)
        for indices in train_test_split(
            np.arange(size), test_size=validation_fraction, random_state=seed, shuffle=True
        )
    )  # type: ignore[return-value]


def kfold_splits(size: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic shuffled K-fold partitions."""
    _validate_split_size(size)
    if not 2 <= k <= size:
        raise ValueError(f"'k' must be between 2 and dataset size ({size})")
    from sklearn.model_selection import KFold

    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [
        (np.asarray(train, dtype=int), np.asarray(valid, dtype=int))
        for train, valid in splitter.split(np.arange(size))
    ]


def validate_transfer_alignment(lf_atoms: list[Any], hf_atoms: list[Any]) -> None:
    """Ensure each LF geometry has the matching HF geometry at its index.

    Equality of atom counts and atomic numbers is mandatory.  Coordinates are
    compared when the atom objects expose them, and stable geometry identifiers
    are compared when both paired objects provide the same supported key.
    """
    if len(lf_atoms) != len(hf_atoms):
        raise ValueError("LF and HF training datasets must contain equally many aligned geometries")
    for index, (lf_atom, hf_atom) in enumerate(zip(lf_atoms, hf_atoms)):
        if len(lf_atom) != len(hf_atom):
            raise ValueError(f"LF/HF geometry {index} has different atom counts")
        if not np.array_equal(_atomic_numbers(lf_atom), _atomic_numbers(hf_atom)):
            raise ValueError(f"LF/HF geometry {index} has different atomic numbers")
        _validate_geometry_identifier(lf_atom, hf_atom, index)
        _validate_coordinates(lf_atom, hf_atom, index)


def selected_transfer_atoms(
    lf_atoms: list[Any], hf_atoms: list[Any], selected_lf_indices: Iterable[int]
) -> tuple[list[Any], list[Any], np.ndarray]:
    """Map LF selector indices to their validated aligned LF/HF geometries."""
    validate_transfer_alignment(lf_atoms, hf_atoms)
    raw_indices = list(selected_lf_indices)
    if not raw_indices:
        raise ValueError("Transfer selection must contain one or more indices")
    if any(isinstance(index, bool) or not isinstance(index, (int, np.integer)) for index in raw_indices):
        raise ValueError("Transfer selection indices must be integers")
    indices = np.asarray(raw_indices, dtype=int)
    if np.any(indices < 0) or np.any(indices >= len(lf_atoms)):
        raise ValueError("Transfer selection contains an out-of-range LF index")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Transfer selection contains duplicate LF indices")
    return (
        [lf_atoms[index] for index in indices],
        [hf_atoms[index] for index in indices],
        indices,
    )


def resolved_e0s(builder: Any) -> dict[str, float]:
    """Return the fitted or configured atomic-energy offsets from metadata."""
    from ase.data import chemical_symbols

    metadata = builder.get_metadata()
    return {
        chemical_symbols[int(atomic_number)]: float(energy)
        for atomic_number, energy in zip(
            metadata.atomic_numbers, metadata.atomic_energies
        )
    }


def _validate_split_size(size: int) -> None:
    if size < 2:
        raise ValueError("At least two geometries are required to make a validation split")


def _atomic_numbers(atoms: Any) -> np.ndarray:
    if hasattr(atoms, "get_atomic_numbers"):
        return np.asarray(atoms.get_atomic_numbers(), dtype=int)
    if hasattr(atoms, "numbers"):
        return np.asarray(atoms.numbers, dtype=int)
    raise ValueError("Cannot read atomic numbers while validating LF/HF alignment")


def _validate_coordinates(lf_atom: Any, hf_atom: Any, index: int) -> None:
    try:
        lf_coordinates = _coordinates(lf_atom)
        hf_coordinates = _coordinates(hf_atom)
    except (AttributeError, TypeError, ValueError):
        return
    if lf_coordinates is not None and hf_coordinates is not None and not np.allclose(
        lf_coordinates, hf_coordinates, rtol=1.0e-7, atol=1.0e-8
    ):
        raise ValueError(f"LF/HF geometry {index} has different coordinates")


def _coordinates(atoms: Any) -> np.ndarray | None:
    if hasattr(atoms, "get_positions"):
        return np.asarray(atoms.get_positions(), dtype=float)
    if hasattr(atoms, "positions"):
        return np.asarray(atoms.positions, dtype=float)
    return None


def _validate_geometry_identifier(lf_atom: Any, hf_atom: Any, index: int) -> None:
    lf_info = getattr(lf_atom, "info", {})
    hf_info = getattr(hf_atom, "info", {})
    if not isinstance(lf_info, dict) or not isinstance(hf_info, dict):
        return
    for key in ("geometry_id", "geometry_uuid", "uuid"):
        if key in lf_info and key in hf_info and lf_info[key] != hf_info[key]:
            raise ValueError(f"LF/HF geometry {index} has different '{key}' values")
