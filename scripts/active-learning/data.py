"""HF-grid access with an explicit boundary between labels and inference."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.data import read_atoms
from sampling_methods import descriptors


@dataclass
class HFGrid:
    """A labeled HF grid that reveals targets only for acquired indices."""

    _labelled_atoms: list[Any]
    grid_shape: tuple[int, int]
    energy_key: str
    forces_key: str
    _coordinates: np.ndarray

    @classmethod
    def from_xyz(
        cls,
        path: Path,
        *,
        grid_shape: tuple[int, int],
        energy_key: str,
        forces_key: str,
    ) -> "HFGrid":
        """Load a complete labeled HF grid and validate its topology."""
        atoms = read_atoms(path)
        expected_count = int(np.prod(grid_shape))
        if len(atoms) != expected_count:
            raise ValueError(
                "'grid_shape' expects "
                f"{expected_count} geometries but HF grid contains {len(atoms)}"
            )
        for index, atom in enumerate(atoms):
            missing = [
                key
                for key in (energy_key, forces_key)
                if key not in atom.info and key not in atom.arrays
            ]
            if missing:
                names = ", ".join(repr(key) for key in missing)
                raise ValueError(
                    f"HF grid geometry {index} is missing required target(s): {names}"
                )

        coordinates = np.asarray(
            [
                (
                    float(descriptors.get_descriptor("bond_lengths", atom)[0]),
                    float(descriptors.get_descriptor("dihedral", atom)[0]),
                )
                for atom in atoms
            ],
            dtype=float,
        )
        if not np.isfinite(coordinates).all():
            raise ValueError("HF-grid CC bond lengths and dihedrals must be finite")
        return cls(
            _labelled_atoms=atoms,
            grid_shape=grid_shape,
            energy_key=energy_key,
            forces_key=forces_key,
            _coordinates=coordinates,
        )

    @property
    def size(self) -> int:
        """Return the total number of HF-grid geometries."""
        return len(self._labelled_atoms)

    @property
    def coordinates(self) -> np.ndarray:
        """Return CC bond-length and dihedral coordinates by global index."""
        return self._coordinates.copy()

    def reveal(self, indices: Iterable[int]) -> list[Any]:
        """Return independent, labeled atom copies for acquired grid indices."""
        return [deepcopy(self._labelled_atoms[index]) for index in self._indices(indices)]

    def prediction_atoms(self, indices: Iterable[int]) -> list[Any]:
        """Return independent target-free copies for committee prediction.

        X-MACE treats absent targets as zero-weight quantities. Removing both
        configured keys prevents a prediction loader from receiving hidden HF
        energies or forces while preserving only geometry and non-target data.
        """
        result = []
        for index in self._indices(indices):
            atom = deepcopy(self._labelled_atoms[index])
            atom.info.pop(self.energy_key, None)
            atom.info.pop(self.forces_key, None)
            atom.arrays.pop(self.energy_key, None)
            atom.arrays.pop(self.forces_key, None)
            result.append(atom)
        return result

    def unacquired_indices(self, acquired_indices: Iterable[int]) -> np.ndarray:
        """Return ordered global indices absent from the acquired set."""
        acquired = set(self._indices(acquired_indices))
        return np.asarray(
            [index for index in range(self.size) if index not in acquired],
            dtype=int,
        )

    def _indices(self, indices: Iterable[int]) -> list[int]:
        resolved = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                raise TypeError("Grid indices must be integers")
            index = int(index)
            if not 0 <= index < self.size:
                raise IndexError(
                    f"Grid index {index} is outside [0, {self.size - 1}]"
                )
            resolved.append(index)
        if len(set(resolved)) != len(resolved):
            raise ValueError("Grid indices must be unique")
        return resolved
