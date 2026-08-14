"""Tests for the subprocess-orchestrated initial-condition workflow."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sharc_initconds import run_initconds


class RunInitcondsTests(unittest.TestCase):
    def test_validation_rejects_non_xyz_before_model_or_sharc_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "molecule.txt"
            input_path.write_text("not xyz", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "XYZ"):
                run_initconds.validate_setup(input_path)

    def test_validation_requires_model_placeholders_to_be_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "molecule.xyz"
            input_path.write_text("1\nH\nH 0 0 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ENERGY_MODEL and OSC_MODEL"):
                run_initconds.validate_setup(input_path)

    def test_validation_requires_sharc_after_models_are_configured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "molecule.xyz"
            energy_model = root / "energy.model"
            osc_model = root / "osc.model"
            input_path.write_text("1\nH\nH 0 0 0\n", encoding="utf-8")
            energy_model.write_text("", encoding="utf-8")
            osc_model.write_text("", encoding="utf-8")
            with (
                patch.object(run_initconds, "ENERGY_MODEL", str(energy_model)),
                patch.object(run_initconds, "OSC_MODEL", str(osc_model)),
                patch.dict(os.environ, {}, clear=True),
                self.assertRaisesRegex(ValueError, "SHARC is not set"),
            ):
                run_initconds.validate_setup(input_path)

    def test_resolve_excite_accepts_sharc_root_and_bin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            (bin_dir / "driver.py").write_text("", encoding="utf-8")
            excite = bin_dir / "excite.py"
            excite.write_text("", encoding="utf-8")
            self.assertEqual(run_initconds.resolve_excite(str(root)), excite)
            self.assertEqual(run_initconds.resolve_excite(str(bin_dir)), excite)

    def test_workflow_runs_six_commands_with_logs_and_excite_stdin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "molecule.xyz"
            input_path.write_text("1\nH\nH 0 0 0\n", encoding="utf-8")
            energy_model = root / "energy.model"
            osc_model = root / "osc.model"
            excite = root / "excite.py"
            for path in (energy_model, osc_model, excite):
                path.write_text("", encoding="utf-8")

            def fake_run(command, **kwargs):
                cwd = Path(kwargs["cwd"])
                if len(calls) == 4:
                    (cwd / "excite_inp.txt").write_text("excite input", encoding="utf-8")
                if len(calls) == 5:
                    (cwd / "initconds.excited").write_text(
                        "SHARC Initial conditions file, version 4.0   <Excited>\n",
                        encoding="utf-8",
                    )
                stdin_contents.append(None if kwargs["stdin"] is None else kwargs["stdin"].read())
                calls.append((command, kwargs))
                return subprocess.CompletedProcess(command, 0)

            calls = []
            stdin_contents = []
            with (
                patch.object(run_initconds, "ENERGY_MODEL", str(energy_model)),
                patch.object(run_initconds, "OSC_MODEL", str(osc_model)),
                patch.object(run_initconds, "resolve_excite", return_value=excite),
                patch("sharc_initconds.run_initconds.subprocess.run", side_effect=fake_run),
            ):
                output = run_initconds.run_workflow(input_path)

            run_dir = root / "molecule_initconds"
            self.assertEqual(output, run_dir / "initconds.excited")
            self.assertEqual(len(calls), 6)
            self.assertEqual(calls[0][0], [sys.executable, str(run_initconds.SCRIPT_DIR / "xtb_md.py"), str(input_path)])
            self.assertIn(str(energy_model), calls[1][0])
            self.assertIn(str(osc_model), calls[2][0])
            self.assertEqual(calls[3][0][-4:], ["--n-states", "3", "--n-osc", "2"])
            self.assertEqual(calls[4][0], [sys.executable, str(run_initconds.SCRIPT_DIR / "write_initconds-excited.py")])
            self.assertEqual(calls[5][0], [sys.executable, str(excite)])
            self.assertEqual([Path(call[1]["cwd"]) for call in calls], [run_dir] * 6)
            self.assertEqual(
                [Path(call[1]["stdout"].name).name for call in calls],
                ["01_xtb_md.log", "02_energies.log", "03_oscillator_strengths.log", "04_initconds.log", "05_excite_input.log", "06_excite.log"],
            )
            self.assertIsNone(calls[0][1]["stdin"])
            self.assertEqual(stdin_contents[5], "excite input")

    def test_existing_run_directory_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "molecule.xyz"
            input_path.write_text("1\nH\nH 0 0 0\n", encoding="utf-8")
            (root / "molecule_initconds").mkdir()
            with patch.object(run_initconds, "validate_setup", return_value=(root / "energy", root / "osc", root / "excite")):
                with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                    run_initconds.run_workflow(input_path)

    def test_stage_help_and_argument_validation_do_not_import_scientific_dependencies(self) -> None:
        scripts = [
            "xtb_md.py",
            "write_md_traj_with-energies.py",
            "write_md_traj_with-energies-and-osc.py",
            "write_initconds.py",
            "write_initconds-excited.py",
        ]
        for script in scripts:
            result = subprocess.run(
                [sys.executable, str(run_initconds.SCRIPT_DIR / script), "--help"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, script)
        invalid = subprocess.run(
            [sys.executable, str(run_initconds.SCRIPT_DIR / "write_initconds.py"), "--n-states", "3", "--n-osc", "1"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("--n-osc", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
