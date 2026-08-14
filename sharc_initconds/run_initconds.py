#!/usr/bin/env python3
"""Run the SHARC initial-condition workflow from one XYZ structure.

Replace the two EDIT_ME model paths below, set SHARC to a SHARC installation
root (or its bin directory), then run ``python run_initconds.py molecule.xyz``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# EDIT_ME: replace these values with trained X-MACE model paths.
ENERGY_MODEL = "/home/lim_yt/X-MACE-sampling/outputs/base_models/base_model_run_33_fold_3.pt"
OSC_MODEL = "/home/lim_yt/X-MACE-sampling/outputs/base_models/base_model_run_33_fold_3.pt"

N_STATES = 3
N_OSC = 2
SCRIPT_DIR = Path(__file__).resolve().parent


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith("EDIT_ME:") or value.startswith("/path/to/")


def resolve_excite(sharc: str | None) -> Path:
    """Return SHARC's excite.py from a SHARC root or bin directory."""
    if not sharc:
        raise ValueError("SHARC is not set. Set it to the SHARC root or bin directory.")
    supplied = Path(sharc).expanduser()
    bin_dir = supplied if (supplied / "driver.py").is_file() else supplied / "bin"
    if not (bin_dir / "driver.py").is_file():
        raise ValueError(f"SHARC does not contain bin/driver.py: {supplied}")
    excite = bin_dir / "excite.py"
    if not excite.is_file():
        raise ValueError(f"SHARC does not contain excite.py: {excite}")
    return excite.resolve()


def validate_setup(input_path: Path) -> tuple[Path, Path, Path]:
    if input_path.suffix.lower() != ".xyz":
        raise ValueError(f"Input must be an XYZ file: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input XYZ file does not exist: {input_path}")
    if _is_placeholder(ENERGY_MODEL) or _is_placeholder(OSC_MODEL):
        raise ValueError(
            "Set ENERGY_MODEL and OSC_MODEL at the top of "
            "sharc_initconds/run_initconds.py before running."
        )
    energy_model = Path(ENERGY_MODEL).expanduser()
    osc_model = Path(OSC_MODEL).expanduser()
    if not energy_model.is_file():
        raise ValueError(f"ENERGY_MODEL does not exist: {energy_model}")
    if not osc_model.is_file():
        raise ValueError(f"OSC_MODEL does not exist: {osc_model}")
    return energy_model.resolve(), osc_model.resolve(), resolve_excite(os.environ.get("SHARC"))


def run_stage(command: list[str], run_dir: Path, log_name: str, *, stdin=None) -> None:
    """Run one stage in its run directory and retain combined output in a log."""
    with (run_dir / log_name).open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=run_dir,
            stdin=stdin,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _validate_final_output(run_dir: Path) -> Path:
    output = run_dir / "initconds.excited"
    if not output.is_file() or not output.read_text(encoding="utf-8").strip():
        raise RuntimeError("SHARC excite.py did not produce a non-empty initconds.excited file.")
    if not output.read_text(encoding="utf-8").startswith("SHARC Initial conditions file"):
        raise RuntimeError("initconds.excited does not have a SHARC initial-conditions header.")
    return output


def run_workflow(input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    energy_model, osc_model, excite = validate_setup(input_path)
    run_dir = input_path.with_name(f"{input_path.stem}_initconds")
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir()

    stages = [
        ([sys.executable, str(SCRIPT_DIR / "xtb_md.py"), str(input_path)], "01_xtb_md.log"),
        (
            [
                sys.executable,
                str(SCRIPT_DIR / "write_md_traj_with-energies.py"),
                "--model",
                str(energy_model),
                "--n-states",
                str(N_STATES),
            ],
            "02_energies.log",
        ),
        (
            [
                sys.executable,
                str(SCRIPT_DIR / "write_md_traj_with-energies-and-osc.py"),
                "--model",
                str(osc_model),
                "--n-osc",
                str(N_OSC),
            ],
            "03_oscillator_strengths.log",
        ),
        (
            [
                sys.executable,
                str(SCRIPT_DIR / "write_initconds.py"),
                "--n-states",
                str(N_STATES),
                "--n-osc",
                str(N_OSC),
            ],
            "04_initconds.log",
        ),
        ([sys.executable, str(SCRIPT_DIR / "write_initconds-excited.py")], "05_excite_input.log"),
    ]
    for command, log_name in stages:
        run_stage(command, run_dir, log_name)

    with (run_dir / "excite_inp.txt").open("r", encoding="utf-8") as excite_input:
        run_stage([sys.executable, str(excite)], run_dir, "06_excite.log", stdin=excite_input)
    return _validate_final_output(run_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_xyz", type=Path, help="single-geometry XYZ input file")
    args = parser.parse_args(argv)
    try:
        output = run_workflow(args.input_xyz)
    except (ValueError, FileExistsError, subprocess.CalledProcessError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Written {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
