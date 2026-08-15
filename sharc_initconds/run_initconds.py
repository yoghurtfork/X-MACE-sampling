#!/usr/bin/env python3
"""Run the SHARC initial-condition workflow from one XYZ structure.

Set model paths below, set SHARC to a SHARC installation
root (or its bin directory), then run ``python run_initconds.py molecule.xyz``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

# Paths are relative to this script, so the repository can be moved or cloned
# anywhere. Replace these with other relative model paths if needed.
ENERGY_MODEL = "../outputs/base_models/base_model_run_33_fold_3.pt"
OSC_MODEL = "../outputs/base_models/ethene_oscillator_strength.model"

N_STATES = 2
N_OSC = 1
EWIN_LOW = 1.6
EWIN_HIGH = 3.3
TEMPERATURE = 300
MD_STEPS = 1010
MD_TIMESTEP_FS = 1
SAVE_INTERVAL = 10


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
    energy_model = (SCRIPT_DIR / Path(ENERGY_MODEL).expanduser()).resolve()
    osc_model = (SCRIPT_DIR / Path(OSC_MODEL).expanduser()).resolve()
    if not energy_model.is_file():
        raise ValueError(f"ENERGY_MODEL does not exist: {energy_model}")
    if not osc_model.is_file():
        raise ValueError(f"OSC_MODEL does not exist: {osc_model}")
    return energy_model, osc_model, resolve_excite(os.environ.get("SHARC"))


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


def run_workflow(
    input_path: Path,
    *,
    n_states: int = N_STATES,
    n_osc: int = N_OSC,
    ewin_low: float = EWIN_LOW,
    ewin_high: float = EWIN_HIGH,
    temperature: float = TEMPERATURE,
    md_steps: int = MD_STEPS,
    md_timestep_fs: float = MD_TIMESTEP_FS,
    save_interval: int = SAVE_INTERVAL,
) -> Path:
    input_path = input_path.expanduser().resolve()
    energy_model, osc_model, excite = validate_setup(input_path)
    run_dir = input_path.with_name(f"{input_path.stem}_initconds")
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir()

    stages = [
        (
            [
                sys.executable,
                str(SCRIPT_DIR / "xtb_md.py"),
                str(input_path),
                "--temperature",
                str(temperature),
                "--md-steps",
                str(md_steps),
                "--md-timestep-fs",
                str(md_timestep_fs),
                "--save-interval",
                str(save_interval),
            ],
            "01_xtb_md.log",
        ),
        (
            [
                sys.executable,
                str(SCRIPT_DIR / "write_md_traj_with-energies.py"),
                "--model",
                str(energy_model),
                "--n-states",
                str(n_states),
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
                str(n_osc),
            ],
            "03_oscillator_strengths.log",
        ),
        (
            [
                sys.executable,
                str(SCRIPT_DIR / "write_initconds.py"),
                "--n-states",
                str(n_states),
                "--n-osc",
                str(n_osc),
            ],
            "04_initconds.log",
        ),
        (
            [
                sys.executable,
                str(SCRIPT_DIR / "write_initconds-excited.py"),
                "--ewin-low",
                str(ewin_low),
                "--ewin-high",
                str(ewin_high),
            ],
            "05_excite_input.log",
        ),
    ]
    for command, log_name in stages:
        run_stage(command, run_dir, log_name)

    with (run_dir / "excite_inp.txt").open("r", encoding="utf-8") as excite_input:
        run_stage([sys.executable, str(excite)], run_dir, "06_excite.log", stdin=excite_input)
    return _validate_final_output(run_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_xyz", type=Path, help="single-geometry XYZ input file")
    parser.add_argument(
        "--n-states", type=int, default=N_STATES,
        help=f"number of electronic states (default: {N_STATES})",
    )
    parser.add_argument(
        "--n-osc", type=int, default=N_OSC,
        help=f"number of oscillator strengths (default: {N_OSC})",
    )
    parser.add_argument(
        "--ewin-low", type=float, default=EWIN_LOW,
        help=f"lower excitation-window bound in eV (default: {EWIN_LOW})",
    )
    parser.add_argument(
        "--ewin-high", type=float, default=EWIN_HIGH,
        help=f"upper excitation-window bound in eV (default: {EWIN_HIGH})",
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help=f"initial MD temperature in K (default: {TEMPERATURE})",
    )
    parser.add_argument(
        "--md-steps", type=int, default=MD_STEPS,
        help=f"number of MD steps (default: {MD_STEPS})",
    )
    parser.add_argument(
        "--md-timestep-fs", type=float, default=MD_TIMESTEP_FS,
        help=f"MD timestep in fs (default: {MD_TIMESTEP_FS})",
    )
    parser.add_argument(
        "--save-interval", type=int, default=SAVE_INTERVAL,
        help=f"trajectory save interval in MD steps (default: {SAVE_INTERVAL})",
    )
    args = parser.parse_args(argv)
    if args.n_states < 2 or args.n_osc != args.n_states - 1:
        parser.error("--n-states must be >= 2 and --n-osc must equal --n-states minus one")
    if args.ewin_low >= args.ewin_high:
        parser.error("--ewin-low must be lower than --ewin-high")
    if args.temperature <= 0 or args.md_steps < 1 or args.md_timestep_fs <= 0 or args.save_interval < 1:
        parser.error("--temperature, --md-timestep-fs, and --save-interval must be positive; --md-steps must be at least 1")
    try:
        output = run_workflow(
            args.input_xyz,
            n_states=args.n_states,
            n_osc=args.n_osc,
            ewin_low=args.ewin_low,
            ewin_high=args.ewin_high,
            temperature=args.temperature,
            md_steps=args.md_steps,
            md_timestep_fs=args.md_timestep_fs,
            save_interval=args.save_interval,
        )
    except (ValueError, FileExistsError, subprocess.CalledProcessError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Written {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
