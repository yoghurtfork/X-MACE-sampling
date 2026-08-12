"""Regression tests for variable-length electronic-state data."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import msgpack
import numpy as np

from sampling_methods.descriptors import get_ci_score
from scripts.helper import _maes_by_state
from scripts.read_azoflip_data import write_azobenzene_xyz


class EnergyLevelTests(unittest.TestCase):
    def test_maes_are_labelled_for_every_returned_state(self) -> None:
        self.assertEqual(
            _maes_by_state([0.1, 0.2, 0.3, 0.4], "energy"),
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
