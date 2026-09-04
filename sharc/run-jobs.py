#!/usr/bin/env python3
"""Discover and validate JSON-scheduled SHARC jobs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


EXCLUDED_DIRECTORY_NAMES = {"traj-allmols", "__pycache__", ".git"}
ALLOWED_KEYS = {
    "ignore",
    "mode",
    "reset",
    "model_file",
    "device",
    "energy_unit",
    "distance_unit",
    "tmax_fs",
    "stepsize_fs",
    "nstates",
    "charge",
    "seed",
}
DEFAULTS = {
    "ignore": False,
    "mode": "run",
    "reset": False,
    "device": "cuda",
    "energy_unit": "eV",
    "distance_unit": "Ang",
    "tmax_fs": 10.0,
    "stepsize_fs": 0.5,
    "charge": [0, 0, 0],
    "seed": 42,
}


@dataclass(frozen=True)
class JobConfig:
    job_dir: Path
    relative_path: Path
    values: dict[str, object]


def discover_jobs(sharc_dir: Path) -> list[Path]:
    """Return job directories containing both input.json and initconds.excited."""
    jobs: list[Path] = []
    for input_json in sharc_dir.rglob("input.json"):
        relative_parts = input_json.relative_to(sharc_dir).parts
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative_parts):
            continue
        job_dir = input_json.parent
        if (job_dir / "initconds.excited").is_file():
            jobs.append(job_dir)
    return sorted(jobs, key=lambda job_dir: job_dir.relative_to(sharc_dir).as_posix())


def is_finite_positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def is_integer_list(value: object, *, allow_negative: bool) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(
        isinstance(item, int)
        and not isinstance(item, bool)
        and (allow_negative or item >= 0)
        for item in value
    )


def is_single_token(value: object) -> bool:
    return isinstance(value, str) and bool(value) and not any(char.isspace() for char in value)


def validate_job(input_json: Path, sharc_dir: Path) -> tuple[JobConfig | None, list[str], bool]:
    """Return a validated job, its errors, and whether it was explicitly ignored."""
    relative_path = input_json.parent.relative_to(sharc_dir)
    label = relative_path.as_posix()
    try:
        raw = json.loads(input_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"{label}: invalid JSON: {error}"], False

    if not isinstance(raw, dict):
        return None, [f"{label}: top level must be a JSON object"], False

    errors = [f"{label}: unknown key: {key}" for key in sorted(set(raw) - ALLOWED_KEYS)]
    ignore = raw.get("ignore", False)
    if not isinstance(ignore, bool):
        errors.append(f"{label}: ignore must be a Boolean")
        return None, errors, False
    if ignore:
        return None, errors, not errors

    values = {**DEFAULTS, **raw}
    mode = values["mode"]
    if not isinstance(mode, str) or mode not in {"run", "prepare-only", "run-existing"}:
        errors.append(f"{label}: mode must be run, prepare-only, or run-existing")

    if not isinstance(values["reset"], bool):
        errors.append(f"{label}: reset must be a Boolean")
    if not is_finite_positive_number(values["tmax_fs"]):
        errors.append(f"{label}: tmax_fs must be a finite positive number")
    if not is_finite_positive_number(values["stepsize_fs"]):
        errors.append(f"{label}: stepsize_fs must be a finite positive number")
    if (
        is_finite_positive_number(values["tmax_fs"])
        and is_finite_positive_number(values["stepsize_fs"])
        and values["stepsize_fs"] > values["tmax_fs"]
    ):
        errors.append(f"{label}: stepsize_fs must be no greater than tmax_fs")
    if not isinstance(values["seed"], int) or isinstance(values["seed"], bool):
        errors.append(f"{label}: seed must be an integer")
    if "nstates" in values and not is_integer_list(values["nstates"], allow_negative=False):
        errors.append(f"{label}: nstates must be a non-empty array of non-negative integers")
    if not is_integer_list(values["charge"], allow_negative=True):
        errors.append(f"{label}: charge must be a non-empty array of integers")
    for key in ("device", "energy_unit", "distance_unit"):
        if not is_single_token(values[key]):
            errors.append(f"{label}: {key} must be a non-empty single-token string")

    if mode == "run-existing":
        forbidden = sorted(set(raw) - {"ignore", "mode"})
        for key in forbidden:
            errors.append(f"{label}: {key} is not allowed with mode run-existing")
    elif "model_file" not in raw:
        errors.append(f"{label}: model_file is required with mode {mode}")
    elif not isinstance(raw["model_file"], str) or not raw["model_file"]:
        errors.append(f"{label}: model_file must be a non-empty path string")
    else:
        model_file = (input_json.parent / raw["model_file"]).resolve()
        if not model_file.is_file():
            errors.append(f"{label}: model_file must name an existing regular file: {model_file}")
        else:
            values["model_file"] = str(model_file)

    if errors:
        return None, errors, False
    return JobConfig(input_json.parent, relative_path, values), [], False


def runner_arguments(job: JobConfig, sharc_dir: Path, gpu_ids: str | None) -> list[str]:
    """Translate one validated JSON job to the shell runner's arguments."""
    values = job.values
    mode = values["mode"]
    args = ["bash", str(sharc_dir / "run-local-ensemble.sh"), job.relative_path.as_posix()]

    if mode == "prepare-only":
        args.append("--prepare-only")
    elif mode == "run-existing":
        args.append("--run-existing")

    if mode != "run-existing":
        if values["reset"]:
            args.append("--reset")
        args.extend(
            [
                "--model-file",
                str(values["model_file"]),
                "--device",
                str(values["device"]),
                "--energy-unit",
                str(values["energy_unit"]),
                "--distance-unit",
                str(values["distance_unit"]),
                "--tmax-fs",
                str(values["tmax_fs"]),
                "--stepsize-fs",
                str(values["stepsize_fs"]),
                "--charge",
                " ".join(str(value) for value in values["charge"]),
                "--seed",
                str(values["seed"]),
            ]
        )
        if "nstates" in values:
            args.extend(("--nstates", " ".join(str(value) for value in values["nstates"])))

    if gpu_ids is not None and mode != "prepare-only":
        args.extend(("--gpu-ids", gpu_ids))
    return args


def parse_gpu_ids(arguments: list[str]) -> tuple[str | None, str | None]:
    """Parse the optional global GPU-ID list and return an error, if any."""
    if not arguments:
        return None, None
    if len(arguments) != 2 or arguments[0] != "--gpu-ids":
        return None, "Usage: python sharc/run-jobs.py [--gpu-ids 0,1,2,3]"

    gpu_ids = arguments[1]
    if not re.fullmatch(r"\d+(?:,\d+)*", gpu_ids):
        return None, "--gpu-ids must be unique comma-separated non-negative integers, e.g. 0,1."
    parsed_ids = [int(value) for value in gpu_ids.split(",")]
    if len(parsed_ids) != len(set(parsed_ids)):
        return None, "--gpu-ids must be unique comma-separated non-negative integers, e.g. 0,1."
    return gpu_ids, None


def main() -> int:
    gpu_ids, argument_error = parse_gpu_ids(sys.argv[1:])
    if argument_error:
        print(argument_error, file=sys.stderr)
        return 2

    sharc_dir = Path(__file__).resolve().parent
    jobs = discover_jobs(sharc_dir)
    if not jobs:
        print("No JSON-scheduled SHARC jobs found.")
        return 0

    enabled_jobs: list[JobConfig] = []
    errors: list[str] = []
    for job_dir in jobs:
        job, job_errors, ignored = validate_job(job_dir / "input.json", sharc_dir)
        errors.extend(job_errors)
        if ignored:
            print(f"Skipping {job_dir.relative_to(sharc_dir).as_posix()}: ignore is true")
        elif job is not None:
            enabled_jobs.append(job)
    if errors:
        print("Invalid JSON-scheduled SHARC jobs:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 2

    if not enabled_jobs:
        return 0
    if not os.environ.get("SHARC"):
        print("Set SHARC to the SHARC bin directory or installation root.", file=sys.stderr)
        return 1

    for job in enabled_jobs:
        try:
            result = subprocess.run(
                runner_arguments(job, sharc_dir, gpu_ids),
                cwd=sharc_dir,
                env=os.environ.copy(),
                shell=False,
                check=False,
            )
        except KeyboardInterrupt:
            print("Interrupted; stopping job queue.", file=sys.stderr)
            return 130
        if result.returncode:
            print(
                f"Job failed: {job.relative_path.as_posix()} (exit code {result.returncode})",
                file=sys.stderr,
            )
            return result.returncode if result.returncode > 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
