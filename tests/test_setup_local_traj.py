"""Regression tests for local SHARC trajectory-file generation."""

from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "sharc" / "setup_local_traj.py"
SPEC = importlib.util.spec_from_file_location("local_setup_traj", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
setup_local_traj = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup_local_traj)


class WriteTrajectoryTests(unittest.TestCase):
    def test_geom_excludes_velocities_for_long_atom_records(self) -> None:
        atom_line = (
            "C 6.0 8.52955949 -1.13534736 0.77542393 12.01100000 "
            "-0.00025000 0.00010000 0.00030000\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "TRAJ_00001"
            qm_shared = root / "QM_shared"
            qm_shared.mkdir()

            setup_local_traj.write_trajectory(
                trajectory,
                [atom_line],
                state=2,
                rngseed=1,
                eref=-572.0,
                nstates="2 0 0",
                charge="0 0 0",
                tmax=30.0,
                stepsize=0.5,
                qm_shared=qm_shared,
            )

            self.assertEqual(
                (trajectory / "geom").read_text(encoding="utf-8"),
                " C   6.0   8.52955949  -1.13534736   0.77542393  12.01100000\n",
            )
            self.assertEqual(
                (trajectory / "veloc").read_text(encoding="utf-8").split(),
                ["-0.00025000", "0.00010000", "0.00030000"],
            )


if __name__ == "__main__":
    unittest.main()
