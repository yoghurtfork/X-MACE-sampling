"""Tests for non-interactive SHARC ``excite.py`` input generation."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "sharc_initconds" / "write_initconds-excited.py"


class WriteExcitedInputTests(unittest.TestCase):
    def test_specified_states_write_sharc_option_two_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--specified-states", "2", "3"],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (Path(directory) / "excite_inp.txt").read_text(encoding="utf-8"),
                "\n\n2\n-inf inf\n2 3\n",
            )

    def test_specified_states_reject_non_positive_indices(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--specified-states", "0"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
