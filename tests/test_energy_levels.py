"""Regression tests for variable-length electronic-state data."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import msgpack
import numpy as np

from sampling_methods import descriptors
from sampling_methods.descriptors import get_ci_score
from scripts.evaluation import maes_by_state
from scripts.read_azoflip_data import write_azobenzene_xyz


class EnergyLevelTests(unittest.TestCase):
    @staticmethod
    def _quadratic_surface(include_duplicate: bool = False):
        """Create a 3-by-3 bond/dihedral surface with analytic Hessians."""
        atoms_list = []
        for dihedral_index, dihedral in enumerate((0.0, 90.0, 180.0)):
            theta = dihedral_index / 2
            for bond_index, bond in enumerate((0.0, 1.0, 2.0)):
                bond_coordinate = bond_index / 2
                energies = [
                    theta**2 + bond_coordinate**2,
                    2 * theta**2 + 3 * theta * bond_coordinate + 4 * bond_coordinate**2,
                    3 * theta**2 + 5 * theta * bond_coordinate + 6 * bond_coordinate**2,
                    4 * theta**2 + 7 * theta * bond_coordinate + 8 * bond_coordinate**2,
                ]
                atoms_list.append(
                    type(
                        "SurfacePoint",
                        (),
                        {
                            "bond": bond,
                            "dihedral": dihedral,
                            "info": {"REF_energy": energies},
                        },
                    )()
                )
        if include_duplicate:
            atoms_list.append(atoms_list[0])
        return atoms_list

    @staticmethod
    def _hessian_norm(atoms_list):
        with patch.object(
            descriptors,
            "get_bond_lengths",
            side_effect=lambda atoms: [atoms.bond],
        ), patch.object(
            descriptors,
            "get_dihedral",
            side_effect=lambda atoms: [atoms.dihedral],
        ):
            return descriptors.get_descriptor("hessian_norm", atoms_list)

    def test_hessian_norm_descriptor_matches_quadratic_surfaces(self) -> None:
        descriptor = self._hessian_norm(self._quadratic_surface(include_duplicate=True))

        expected = np.array([np.sqrt(8), np.sqrt(98), np.sqrt(230), np.sqrt(418)])
        self.assertEqual(descriptor.shape, (10, 4))
        np.testing.assert_allclose(descriptor, np.tile(expected, (10, 1)))
        np.testing.assert_allclose(
            self._hessian_norm(self._quadratic_surface()),
            np.tile(expected, (9, 1)),
        )

        single_state_surface = self._quadratic_surface()
        for atoms in single_state_surface:
            atoms.info["REF_energy"] = atoms.info["REF_energy"][:1]
        np.testing.assert_allclose(
            self._hessian_norm(single_state_surface),
            np.full((9, 1), np.sqrt(8)),
        )

    def test_hessian_norm_descriptor_rejects_invalid_grids_and_states(self) -> None:
        complete_surface = self._quadratic_surface()

        with self.assertRaisesRegex(ValueError, "complete rectangular"):
            self._hessian_norm(complete_surface[:-1])
        with self.assertRaisesRegex(ValueError, "at least three unique"):
            self._hessian_norm(complete_surface[:6])

        complete_surface[-1].info["REF_energy"] = [0.0, 1.0]
        with self.assertRaisesRegex(ValueError, "same number"):
            self._hessian_norm(complete_surface)

    def test_maes_are_labelled_for_every_returned_state(self) -> None:
        self.assertEqual(
            maes_by_state([0.1, 0.2, 0.3, 0.4], "energy"),
            {"S0": 0.1, "S1": 0.2, "S2": 0.3, "S3": 0.4},
        )

    def test_ci_score_accepts_any_number_of_adjacent_states(self) -> None:
        atoms = type(
            "Atoms",
            (),
            {
                "info": {"REF_energy": [[0.0, 1.0, 3.0, 6.0]]},
                "arrays": {"REF_forces": np.zeros((2, 4, 3))},
                "__len__": lambda self: 2,
            },
        )()

        score = get_ci_score(atoms)

        self.assertEqual(len(score), 1)
        self.assertAlmostEqual(score[0], 1.0 / 1.000001)

    def test_azobenzene_exporter_writes_every_available_state(self) -> None:
        geometry = {
            "species": {"inchikey": "DMLAVOWQYNRWNQ-YPKPFQOONA-N"},
            "xyz": [[1, 0.0, 0.0, 0.0]],
            "props": {
                "totalenergy": 0.0,
                "forces": [[0.0, 0.0, 0.0]],
                "excitedstates": [
                    {"energy": 1.0, "forces": [[0.1, 0.0, 0.0]]},
                    {"energy": 2.0, "forces": [[0.2, 0.0, 0.0]]},
                    {"energy": 3.0, "forces": [[0.3, 0.0, 0.0]]},
                ],
            },
        }
        with TemporaryDirectory() as temporary_dir:
            input_path = Path(temporary_dir) / "switches.msgpack"
            output_path = Path(temporary_dir) / "states.xyz"
            input_path.write_bytes(msgpack.packb({"geometry": geometry}))

            _, _, written_counts = write_azobenzene_xyz(input_path, output_path)

            self.assertEqual(written_counts, Counter({"Z": 1}))
            self.assertIn('[0.0, 1.0, 2.0, 3.0]', output_path.read_text())


if __name__ == "__main__":
    unittest.main()
