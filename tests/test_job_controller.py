"""Regression tests for multi-GPU job scheduling helpers."""

from __future__ import annotations

import json
import os
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import job_controller


class JobControllerTests(unittest.TestCase):
    def test_parse_gpu_indices_rejects_invalid_selection(self) -> None:
        self.assertEqual(job_controller._parse_gpu_indices(None, 3), [0, 1, 2])
        self.assertEqual(job_controller._parse_gpu_indices("2,0", 3), [2, 0])
        with self.assertRaises(ValueError):
            job_controller._parse_gpu_indices("0,0", 3)
        with self.assertRaises(ValueError):
            job_controller._parse_gpu_indices("3", 3)

    def test_reserve_run_dir_creates_distinct_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            first = job_controller._reserve_run_dir(output_dir)
            second = job_controller._reserve_run_dir(output_dir)
            self.assertEqual(first.name, "run_0")
            self.assertEqual(second.name, "run_1")
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_gpu_worker_sets_visibility_before_running_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run_0"
            run_dir.mkdir()
            job = job_controller.Job(Path("job.json"), False, run_dir)
            jobs: queue.Queue[object] = queue.Queue()
            results: queue.Queue[object] = queue.Queue()
            jobs.put(job)
            jobs.put(None)
            previous = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                with patch.object(
                    job_controller, "_run_job", return_value=run_dir / "result.json"
                ) as run_job:
                    job_controller._gpu_worker(2, jobs, results)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")
                run_job.assert_called_once_with(job)
                self.assertEqual(results.get_nowait()[1], 2)
            finally:
                if previous is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = previous

    def test_cuda_jobs_fail_clearly_when_no_gpu_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "cuda.json").write_text(
                json.dumps({"device": "cuda", "transfer_learning": False}),
                encoding="utf-8",
            )
            with (
                patch("sys.argv", ["job_controller.py", "--input-dir", str(input_dir)]),
                patch.object(job_controller, "_detected_gpu_count", return_value=0),
            ):
                self.assertEqual(job_controller.main(), 1)


if __name__ == "__main__":
    unittest.main()
