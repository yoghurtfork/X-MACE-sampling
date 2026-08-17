"""Tests for the subprocess-orchestrated initial-condition workflow."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
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
            with (
                patch.object(run_initconds, "ENERGY_MODEL", "EDIT_ME: energy model"),
                patch.object(run_initconds, "OSC_MODEL", "EDIT_ME: oscillator model"),
                self.assertRaisesRegex(ValueError, "ENERGY_MODEL"),
            ):
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
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    output = run_initconds.run_workflow(input_path)

            run_dir = root / "molecule_initconds"
            self.assertEqual(output, run_dir / "initconds.excited")
            self.assertEqual(len(calls), 6)
            self.assertEqual(calls[0][0][:3], [sys.executable, str(run_initconds.SCRIPT_DIR / "xtb_md.py"), str(input_path)])
            self.assertIn(str(energy_model), calls[1][0])
            self.assertIn(str(osc_model), calls[2][0])
            self.assertEqual(
                calls[3][0][-4:],
                ["--n-states", str(run_initconds.N_STATES), "--n-osc", str(run_initconds.N_OSC)],
            )
            self.assertEqual(calls[4][0][:2], [sys.executable, str(run_initconds.SCRIPT_DIR / "write_initconds-excited.py")])
            self.assertEqual(calls[5][0], [sys.executable, str(excite)])
            self.assertEqual([Path(call[1]["cwd"]) for call in calls], [run_dir] * 6)
            self.assertEqual(
                [Path(call[1]["stdout"].name).name for call in calls],
                ["01_xtb_md.log", "02_energies.log", "03_oscillator_strengths.log", "04_initconds.log", "05_excite_input.log", "06_excite.log"],
            )
            self.assertIsNone(calls[0][1]["stdin"])
            self.assertEqual(stdin_contents[5], "excite input")
            lines = stdout.getvalue().splitlines()
            labels = [
                "xTB relaxation/MD",
                "energy prediction",
                "oscillator-strength prediction",
                "initconds writing",
                "excitation-input writing",
                "SHARC state selection",
            ]
            log_names = [
                "01_xtb_md.log",
                "02_energies.log",
                "03_oscillator_strengths.log",
                "04_initconds.log",
                "05_excite_input.log",
                "06_excite.log",
            ]
            expected_progress = []
            for stage_number, (label, log_name) in enumerate(zip(labels, log_names), start=1):
                expected_progress.extend([
                    f"Starting stage {stage_number}/6: {label}...",
                    f"Finished stage {stage_number}/6: {label}.",
                    f"Log written to {run_dir / log_name}",
                    "",
                ])
            self.assertEqual(lines, expected_progress)

    def test_specified_states_skips_oscillator_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "molecule.xyz"
            input_path.write_text("1\nH\nH 0 0 0\n", encoding="utf-8")
            energy_model = root / "energy.model"
            excite = root / "excite.py"
            energy_model.write_text("", encoding="utf-8")
            excite.write_text("", encoding="utf-8")
            calls = []
            stdin_contents = []

            def fake_run(command, **kwargs):
                cwd = Path(kwargs["cwd"])
                if len(calls) == 2:
                    (cwd / "excite_inp.txt").write_text(
                        "\n\n2\n-inf inf\n2\n\n", encoding="utf-8"
                    )
                if len(calls) == 3:
                    (cwd / "initconds.excited").write_text(
                        "SHARC Initial conditions file, version 4.0   <Excited>\n",
                        encoding="utf-8",
                    )
                stdin_contents.append(
                    None if kwargs["stdin"] is None else kwargs["stdin"].read()
                )
                calls.append((command, kwargs))
                return subprocess.CompletedProcess(command, 0)

            with (
                patch.object(run_initconds, "ENERGY_MODEL", str(energy_model)),
                patch.object(run_initconds, "OSC_MODEL", "EDIT_ME: unused"),
                patch.object(run_initconds, "resolve_excite", return_value=excite),
                patch("sharc_initconds.run_initconds.subprocess.run", side_effect=fake_run),
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    output = run_initconds.run_workflow(
                        input_path,
                        n_states=2,
                        n_osc=1,
                        specified_states=[2],
                    )

            run_dir = root / "molecule_initconds"
            self.assertEqual(output, run_dir / "initconds.excited")
            self.assertEqual(len(calls), 5)
            self.assertEqual(
                [Path(call[1]["stdout"].name).name for call in calls],
                ["01_xtb_md.log", "02_energies.log", "04_initconds.log", "05_excite_input.log", "06_excite.log"],
            )
            self.assertNotIn("write_md_traj_with-energies-and-osc.py", " ".join(map(str, calls)))
            self.assertIn("--without-oscillator-strengths", calls[2][0])
            self.assertEqual(calls[3][0][-2:], ["--specified-states", "2"])
            self.assertEqual(stdin_contents[4], "\n\n2\n-inf inf\n2\n\n")
            self.assertFalse((run_dir / "03_oscillator_strengths.log").exists())
            lines = stdout.getvalue().splitlines()
            self.assertIn(
                "Skipping stage 3/6: oscillator-strength prediction because initial excited states were specified.",
                lines,
            )
            self.assertLess(
                lines.index("Finished stage 2/6: energy prediction."),
                lines.index("Skipping stage 3/6: oscillator-strength prediction because initial excited states were specified."),
            )

    def test_specified_states_are_validated_before_workflow(self) -> None:
        for states in ([-1], [1], [3]):
            with self.subTest(states=states):
                with self.assertRaisesRegex(ValueError, "--specify-excited-states"):
                    run_initconds.validate_specified_states(states, n_states=2)

    def test_main_rejects_invalid_specified_states_before_setup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "molecule.xyz"
            input_path.write_text("1\nH\nH 0 0 0\n", encoding="utf-8")
            with patch.object(run_initconds, "validate_setup") as setup:
                with self.assertRaises(SystemExit) as error:
                    run_initconds.main([str(input_path), "--specify-excited-states", "1"])
            self.assertEqual(error.exception.code, 2)
            setup.assert_not_called()

    def test_main_overwrites_existing_run_directory_after_yes_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "molecule.xyz"
            input_path.write_text("1\nH\nH 0 0 0\n", encoding="utf-8")
            run_dir = root / "molecule_initconds"
            run_dir.mkdir()
            old_artifact = run_dir / "old.log"
            old_artifact.write_text("old run", encoding="utf-8")
            output = run_dir / "initconds.excited"

            def fake_workflow(path, **kwargs):
                self.assertEqual(path, input_path)
                self.assertFalse(run_dir.exists())
                run_dir.mkdir()
                output.write_text("SHARC Initial conditions file\n", encoding="utf-8")
                return output

            with (
                patch.object(run_initconds, "validate_setup", return_value=(root / "energy", root / "osc", root / "excite")),
                patch.object(run_initconds, "run_workflow", side_effect=fake_workflow),
                patch("builtins.input", return_value="yes"),
            ):
                self.assertEqual(run_initconds.main([str(input_path)]), 0)

            self.assertFalse(old_artifact.exists())
            self.assertTrue(output.is_file())

    def test_main_cancellation_keeps_existing_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "molecule.xyz"
            input_path.write_text("1\nH\nH 0 0 0\n", encoding="utf-8")
            run_dir = root / "molecule_initconds"
            run_dir.mkdir()
            old_artifact = run_dir / "old.log"
            old_artifact.write_text("old run", encoding="utf-8")

            for response in ("n", "", EOFError()):
                with self.subTest(response=response):
                    input_side_effect = response if isinstance(response, BaseException) else None
                    with (
                        patch.object(run_initconds, "validate_setup", return_value=(root / "energy", root / "osc", root / "excite")),
                        patch.object(run_initconds, "run_workflow") as workflow,
                        patch("builtins.input", side_effect=input_side_effect) if input_side_effect else patch("builtins.input", return_value=response),
                    ):
                        self.assertEqual(run_initconds.main([str(input_path)]), 0)
                    workflow.assert_not_called()
                    self.assertEqual(old_artifact.read_text(encoding="utf-8"), "old run")

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
