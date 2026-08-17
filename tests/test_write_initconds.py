"""Tests for SHARC oscillator-strength encoding."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path

from sharc_initconds import write_initconds


class EffectiveTransitionDipoleTests(unittest.TestCase):
    def test_positive_strength_reconstructs_the_input_oscillator_strength(self) -> None:
        gap_ha = 0.25
        oscillator_strength = 0.12
        dipole_x, clamped = write_initconds.effective_transition_dipole(oscillator_strength, gap_ha)

        reconstructed = (2.0 / 3.0) * gap_ha * dipole_x**2
        self.assertFalse(clamped)
        self.assertAlmostEqual(reconstructed, oscillator_strength)

    def test_zero_and_negative_strengths_produce_dark_transitions(self) -> None:
        self.assertEqual(write_initconds.effective_transition_dipole(0.0, 0.25), (0.0, False))
        self.assertEqual(write_initconds.effective_transition_dipole(-0.1, 0.25), (0.0, True))

    def test_invalid_strengths_and_gaps_are_rejected(self) -> None:
        for oscillator_strength, gap_ha in ((math.nan, 0.2), (math.inf, 0.2), (0.1, math.nan), (0.1, 0.0), (0.1, -0.2)):
            with self.subTest(oscillator_strength=oscillator_strength, gap_ha=gap_ha):
                with self.assertRaises(ValueError):
                    write_initconds.effective_transition_dipole(oscillator_strength, gap_ha)


class WriteInitcondsTests(unittest.TestCase):
    def test_writer_places_effective_dipole_in_sharc_state_record(self) -> None:
        from ase import Atoms
        from ase.io import write

        frames = []
        for energies, oscillator_strengths in (([0.0, 2.0, 3.0], [0.1, -0.2]), ([0.0, 2.0, 3.0], [0.1, -0.2])):
            atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
            atoms.set_velocities([[0.0, 0.0, 0.0]])
            atoms.info["REF_energy"] = energies
            atoms.info["REF_osc-strength"] = oscillator_strengths
            frames.append(atoms)

        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            write(workdir / "md_traj_with-energies-and-osc.xyz", frames)
            previous_dir = Path.cwd()
            try:
                os.chdir(workdir)
                write_initconds.run(n_states=3, n_osc=2)
            finally:
                os.chdir(previous_dir)

            state_lines = (workdir / "initconds").read_text(encoding="utf-8").split("States\n", 1)[1].splitlines()
            s1_fields = state_lines[1].split()
            s2_fields = state_lines[2].split()
            expected_dipole = math.sqrt(3.0 * 0.1 / (2.0 * 2.0))
            self.assertAlmostEqual(float(s1_fields[3]), expected_dipole, places=7)
            self.assertEqual(s1_fields[4:9], ["0.00000000"] * 5)
            self.assertEqual(float(s2_fields[3]), 0.0)

    def test_writer_without_oscillator_strengths_accepts_energy_only_frames(self) -> None:
        from ase import Atoms
        from ase.io import write

        frames = []
        for _ in range(2):
            atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
            atoms.set_velocities([[0.0, 0.0, 0.0]])
            atoms.info["REF_energy"] = [0.0, 0.2]
            frames.append(atoms)

        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            write(workdir / "md_traj_with-energies.xyz", frames)
            previous_dir = Path.cwd()
            try:
                os.chdir(workdir)
                write_initconds.run(
                    n_states=2, n_osc=1, without_oscillator_strengths=True
                )
            finally:
                os.chdir(previous_dir)

            state_lines = (workdir / "initconds").read_text(encoding="utf-8").split("States\n", 1)[1].splitlines()
            s1_fields = state_lines[1].split()
            self.assertEqual(s1_fields[3:9], ["0.00000000"] * 6)
            self.assertEqual(s1_fields[10], "0.00000000")


if __name__ == "__main__":
    unittest.main()
