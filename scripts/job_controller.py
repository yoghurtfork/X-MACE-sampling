"""Schedule JSON-configured training jobs across available CUDA devices."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import load_config
from scripts.state import reserve_run_dir

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


class Job(NamedTuple):
    """A job whose output directory has already been reserved."""

    config_path: Path
    run_dir: Path


def _read_job_options(config_path: Path) -> tuple[bool, str]:
    """Validate a job once and return only its scheduling settings."""
    config, _warnings = load_config(config_path)
    return config["ignore"], config["device"]


def _run_job(job: Job) -> Path:
    """Start the single trainer after a worker selects its CUDA device."""
    from scripts import train

    return train.run_config(job.config_path, job.run_dir.parent, job.run_dir)


def _gpu_worker(
    gpu_index: int, jobs: mp.queues.Queue, results: mp.queues.Queue
) -> None:
    """Run sequential jobs with exactly one physical GPU visible to PyTorch."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    while True:
        job = jobs.get()
        if job is None:
            return
        try:
            result_path = _run_job(job)
            results.put((job.config_path, gpu_index, str(result_path), None))
        except Exception as exc:  # report and continue with the next job
            results.put((job.config_path, gpu_index, None, str(exc)))


def _parse_gpu_indices(value: str | None, device_count: int) -> list[int]:
    if device_count < 1:
        raise ValueError("No CUDA devices were detected")
    if value is None:
        return list(range(device_count))
    try:
        indices = [int(part) for part in value.split(",") if part]
    except ValueError as exc:
        raise ValueError("'--gpus' must be a comma-separated list of integers") from exc
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("'--gpus' must contain one or more unique GPU indices")
    invalid = [index for index in indices if index < 0 or index >= device_count]
    if invalid:
        raise ValueError(
            f"GPU indices {invalid} are unavailable; detected 0 through {device_count - 1}"
        )
    return indices


def _detected_gpu_count() -> int:
    # Import only in the scheduler.  GPU workers set CUDA_VISIBLE_DEVICES
    # before they import any module that imports PyTorch.
    import torch

    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _run_gpu_jobs(jobs: list[Job], gpu_indices: list[int], slots: int) -> int:
    """Run queued CUDA jobs in isolated processes, one queue consumer per slot."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    results = context.Queue()
    worker_gpus = [gpu for gpu in gpu_indices for _ in range(slots)]
    workers = [
        context.Process(target=_gpu_worker, args=(gpu, queue, results))
        for gpu in worker_gpus
    ]
    for worker in workers:
        worker.start()
    for job in jobs:
        queue.put(job)
    for _ in workers:
        queue.put(None)

    failed = 0
    for _ in jobs:
        config_path, gpu_index, result_path, error = results.get()
        if error is None:
            print(
                f"GPU {gpu_index}: {config_path.name} completed; "
                f"results written to {result_path}"
            )
        else:
            failed += 1
            print(
                f"GPU {gpu_index}: job failed for {config_path}: {error}",
                file=sys.stderr,
            )
    for worker in workers:
        worker.join()
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gpus",
        help="comma-separated CUDA GPU indices to use (default: all detected)",
    )
    parser.add_argument(
        "--max-gpu-jobs-per-device",
        type=int,
        default=1,
        help="maximum simultaneous jobs per selected GPU (default: 1)",
    )
    args = parser.parse_args()
    if args.max_gpu_jobs_per_device < 1:
        parser.error("'--max-gpu-jobs-per-device' must be positive")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    config_paths = sorted(input_dir.glob("*.json"))
    if not config_paths:
        print(f"No JSON input files found in {input_dir}", file=sys.stderr)
        return 1

    cpu_jobs: list[Job] = []
    cuda_specs: list[Path] = []
    failed = 0
    for config_path in config_paths:
        try:
            ignore, device = _read_job_options(config_path)
            if ignore:
                print(f"Skipping {config_path.name} ('ignore' is true)")
            elif device == "cuda":
                cuda_specs.append(config_path)
            else:
                cpu_jobs.append(Job(config_path, reserve_run_dir(output_dir)))
        except Exception as exc:
            failed += 1
            print(f"Job failed for {config_path}: {exc}", file=sys.stderr)

    for job in cpu_jobs:
        try:
            result_path = _run_job(job)
            print(f"CPU: {job.config_path.name} completed; results written to {result_path}")
        except Exception as exc:
            failed += 1
            print(f"CPU: job failed for {job.config_path}: {exc}", file=sys.stderr)

    if cuda_specs:
        device_count = _detected_gpu_count()
        try:
            gpu_indices = _parse_gpu_indices(args.gpus, device_count)
        except ValueError as exc:
            for config_path in cuda_specs:
                print(f"CUDA: job failed for {config_path}: {exc}", file=sys.stderr)
            failed += len(cuda_specs)
        else:
            cuda_jobs = [
                Job(config_path, reserve_run_dir(output_dir))
                for config_path in cuda_specs
            ]
            print(
                f"Scheduling {len(cuda_jobs)} CUDA job(s) across GPU(s) {gpu_indices} "
                f"with {args.max_gpu_jobs_per_device} worker(s) per GPU"
            )
            failed += _run_gpu_jobs(
                cuda_jobs, gpu_indices, args.max_gpu_jobs_per_device
            )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
