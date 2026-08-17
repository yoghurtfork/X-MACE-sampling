#!/usr/bin/env python3
"""Run the SHARC initial-condition workflow from one XYZ structure.

Set model paths below, set SHARC to a SHARC installation
root (or its bin directory), then run ``python run_initconds.py molecule.xyz``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

# Paths are relative to this script, so the repository can be moved or cloned
# anywhere. Replace these with other relative model paths if needed.
ENERGY_MODEL = "../outputs/base_models/base_model_azoflip.pt"
OSC_MODEL = "../outputs/osc_strength_models/osc_strength_azobenzene.model"

N_STATES = 2
N_OSC = 1
EWIN_LOW = 0
EWIN_HIGH = 20
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


def validate_setup(
    input_path: Path, *, require_oscillator_model: bool = True
) -> tuple[Path, Path | None, Path]:
    if input_path.suffix.lower() != ".xyz":
        raise ValueError(f"Input must be an XYZ file: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input XYZ file does not exist: {input_path}")
    if _is_placeholder(ENERGY_MODEL):
        raise ValueError(
            "Set ENERGY_MODEL at the top of sharc_initconds/run_initconds.py "
            "before running."
        )
    if require_oscillator_model and _is_placeholder(OSC_MODEL):
        raise ValueError(
            "Set ENERGY_MODEL and OSC_MODEL at the top of "
            "sharc_initconds/run_initconds.py before running."
        )
    energy_model = (SCRIPT_DIR / Path(ENERGY_MODEL).expanduser()).resolve()
    osc_model = (
        (SCRIPT_DIR / Path(OSC_MODEL).expanduser()).resolve()
        if require_oscillator_model
        else None
    )
    if not energy_model.is_file():
        raise ValueError(f"ENERGY_MODEL does not exist: {energy_model}")
    if osc_model is not None and not osc_model.is_file():
        raise ValueError(f"OSC_MODEL does not exist: {osc_model}")
    return energy_model, osc_model, resolve_excite(os.environ.get("SHARC"))


def run_stage(
    command: list[str],
    run_dir: Path,
    log_name: str,
    *,
    stage_number: int,
    stage_label: str,
    stdin=None,
) -> None:
    """Run one stage in its run directory and retain combined output in a log."""
    log_path = (run_dir / log_name).resolve()
    print(f"Starting stage {stage_number}/6: {stage_label}...", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=run_dir,
            stdin=stdin,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(
        f"Finished stage {stage_number}/6: {stage_label}.\nLog written to {log_path}\n",
        flush=True,
    )


def _validate_final_output(run_dir: Path) -> Path:
    output = run_dir / "initconds.excited"
    if not output.is_file() or not output.read_text(encoding="utf-8").strip():
        raise RuntimeError("SHARC excite.py did not produce a non-empty initconds.excited file.")
    if not output.read_text(encoding="utf-8").startswith("SHARC Initial conditions file"):
        raise RuntimeError("initconds.excited does not have a SHARC initial-conditions header.")
    return output


def confirm_overwrite(run_dir: Path) -> bool:
    """Ask whether an existing terminal workflow run may be replaced."""
    print(f"Run directory already exists: {run_dir}")
    try:
        answer = input("Overwrite it and continue? [y/n]: ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes", "Y"}


def validate_specified_states(states: list[int] | None, n_states: int) -> None:
    """Validate one-based SHARC excited-state indices, when supplied."""
    if states is None:
        return
    invalid = [state for state in states if state < 2 or state > n_states]
    if invalid:
        raise ValueError(
            "--specify-excited-states values must be excited-state indices "
            f"from 2 through {n_states}; got: {' '.join(map(str, invalid))}"
        )


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
    specified_states: list[int] | None = None,
) -> Path:
    input_path = input_path.expanduser().resolve()
    validate_specified_states(specified_states, n_states)
    energy_model, osc_model, excite = validate_setup(
        input_path, require_oscillator_model=specified_states is None
    )
    run_dir = input_path.with_name(f"{input_path.stem}_initconds")
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir()

    stages = [
        (
            1,
            "xTB relaxation/MD",
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
            2,
            "energy prediction",
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
            4,
            "initconds writing",
            [
                sys.executable,
                str(SCRIPT_DIR / "write_initconds.py"),
                "--n-states",
                str(n_states),
                "--n-osc",
                str(n_osc),
            ] + (["--without-oscillator-strengths"] if specified_states is not None else []),
            "04_initconds.log",
        ),
        (
            5,
            "excitation-input writing",
            [
                sys.executable,
                str(SCRIPT_DIR / "write_initconds-excited.py"),
            ]
            + (
                ["--specified-states", *map(str, specified_states)]
                if specified_states is not None
                else ["--ewin-low", str(ewin_low), "--ewin-high", str(ewin_high)]
            ),
            "05_excite_input.log",
        ),
    ]
    if specified_states is None:
        stages.insert(
            2,
            (
                3,
                "oscillator-strength prediction",
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
        )
    else:
        stages.insert(
            2,
            (
                3,
                "oscillator-strength prediction",
                None,
                None,
            ),
        )

    for stage_number, stage_label, command, log_name in stages:
        if command is None:
            print(
                "Skipping stage 3/6: oscillator-strength prediction because initial "
                "excited states were specified.",
                flush=True,
            )
            continue
        run_stage(
            command,
            run_dir,
            log_name,
            stage_number=stage_number,
            stage_label=stage_label,
        )

    with (run_dir / "excite_inp.txt").open("r", encoding="utf-8") as excite_input:
        run_stage(
            [sys.executable, str(excite)],
            run_dir,
            "06_excite.log",
            stage_number=6,
            stage_label="SHARC state selection",
            stdin=excite_input,
        )
    return _validate_final_output(run_dir)


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("GLIBC_TUNABLES", "glibc.rtld.execstack=2")
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
        "--specify-excited-states",
        type=int,
        nargs="+",
        metavar="STATE",
        help="select SHARC excited-state indices directly instead of using oscillator strengths",
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
    if args.n_states < 2 or (
        args.specify_excited_states is None and args.n_osc != args.n_states - 1
    ):
        parser.error("--n-states must be >= 2 and --n-osc must equal --n-states minus one")
    if args.specify_excited_states is None and args.ewin_low >= args.ewin_high:
        parser.error("--ewin-low must be lower than --ewin-high")
    if args.temperature <= 0 or args.md_steps < 1 or args.md_timestep_fs <= 0 or args.save_interval < 1:
        parser.error("--temperature, --md-timestep-fs, and --save-interval must be positive; --md-steps must be at least 1")
    try:
        input_path = args.input_xyz.expanduser().resolve()
        validate_specified_states(args.specify_excited_states, args.n_states)
        validate_setup(
            input_path,
            require_oscillator_model=args.specify_excited_states is None,
        )
        run_dir = input_path.with_name(f"{input_path.stem}_initconds")
        if run_dir.exists():
            if not run_dir.is_dir():
                raise FileExistsError(f"Run path exists but is not a directory: {run_dir}")
            if not confirm_overwrite(run_dir):
                print(f"Cancelled. Existing run directory left unchanged: {run_dir}")
                return 0
            shutil.rmtree(run_dir)
        output = run_workflow(
            input_path,
            n_states=args.n_states,
            n_osc=args.n_osc,
            ewin_low=args.ewin_low,
            ewin_high=args.ewin_high,
            temperature=args.temperature,
            md_steps=args.md_steps,
            md_timestep_fs=args.md_timestep_fs,
            save_interval=args.save_interval,
            specified_states=args.specify_excited_states,
        )
    except (ValueError, FileExistsError, subprocess.CalledProcessError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Written {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
