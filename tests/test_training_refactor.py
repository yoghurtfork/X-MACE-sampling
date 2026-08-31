"""Regression tests for the unified strict training architecture."""

from __future__ import annotations

import io
import json
import queue
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import data, evaluation, job_controller, state, train
from scripts.config import ConfigError, load_config


def _lf_config(**overrides: object) -> dict[str, object]:
    return {
        "mode": "lf",
        "lf_xyz": "train.xyz",
        "lf_test_xyz": ["test.xyz"],
        **overrides,
    }


class ConfigTests(unittest.TestCase):
    def _load(self, value: dict[str, object]):
        directory = Path(tempfile.mkdtemp())
        path = directory / "run.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return load_config(path)

    def test_normalizes_test_paths_and_defaults(self) -> None:
        config, warnings = self._load(_lf_config(lf_test_xyz="test.xyz"))
        self.assertEqual(config["lf_test_xyz"], [config["lf_xyz"].with_name("test.xyz")])
        self.assertEqual(warnings, [])
        self.assertIn("generate_plots", config)

    def test_unknown_and_legacy_fields_fail_before_training(self) -> None:
        for key in ("unknown", "base_xyz", "transfer_learning"):
            with self.assertRaisesRegex(ConfigError, "Unknown configuration"):
                self._load(_lf_config(**{key: True}))

    def test_transfer_warns_for_present_unused_fields(self) -> None:
        config, warnings = self._load({
            "mode": "transfer", "lf_xyz": "lf.xyz", "hf_xyz": "hf.xyz",
            "hf_test_xyz": "hf-test.xyz", "pretrained_model_path": "lf.pt",
            "descriptor": "energies", "selector": "random_sampling", "n_samples": 2,
            "foundation_model": "ani500k", "lf_test_xyz": "unused.xyz",
            "verbose": False,
        })
        self.assertFalse(config["verbose"])
        self.assertIn("Ignoring 'foundation_model' because it is not used in 'transfer' mode", warnings)
        self.assertIn("Ignoring 'lf_test_xyz' because it is not used in 'transfer' mode", warnings)

    def test_cross_validation_requires_k_and_ignores_fixed_fraction(self) -> None:
        with self.assertRaisesRegex(ConfigError, "'k' is required"):
            self._load(_lf_config(cross_validation=True))
        _, warnings = self._load(_lf_config(cross_validation=True, k=2, validation_fraction=0.2))
        self.assertIn("Ignoring 'validation_fraction' because cross-validation is enabled", warnings)


class DataTests(unittest.TestCase):
    class Atom:
        def __init__(self, number: int = 6, coordinate: float = 0.0) -> None:
            self.numbers = np.array([number, 1])
            self.positions = np.array([[coordinate, 0.0, 0.0], [1.0, 0.0, 0.0]])
            self.info = {"geometry_id": coordinate}

        def __len__(self) -> int:
            return 2

    def test_fixed_and_kfold_splits_are_deterministic(self) -> None:
        first = data.fixed_split(10, 0.2, 42)
        second = data.fixed_split(10, 0.2, 42)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(len(data.kfold_splits(10, 5, 42)), 5)

    def test_transfer_selection_preserves_lf_to_hf_indices(self) -> None:
        lf = [self.Atom(coordinate=float(index)) for index in range(3)]
        hf = [self.Atom(coordinate=float(index)) for index in range(3)]
        selected_lf, selected_hf, indices = data.selected_transfer_atoms(lf, hf, [2, 0])
        self.assertEqual(indices.tolist(), [2, 0])
        self.assertIs(selected_lf[0], lf[2])
        self.assertIs(selected_hf[0], hf[2])
        hf[1].numbers[0] = 8
        with self.assertRaisesRegex(ValueError, "atomic numbers"):
            data.validate_transfer_alignment(lf, hf)


class StateAndEvaluationTests(unittest.TestCase):
    def test_result_json_keeps_scalar_arrays_compact_and_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = state.reserve_run_dir(root)
            result = state.RunState(run, root / "run.json", {"mode": "lf"}, [])
            result.update_stage("lf", {"history": {"train_loss": [0.81, 0.63]}})
            result.fail(RuntimeError("training failed"))
            text = (run / "result.json").read_text(encoding="utf-8")
            self.assertIn('"train_loss": [0.81, 0.63]', text)
            parsed = json.loads(text)
            self.assertEqual(parsed["status"], "failed")
            self.assertEqual(parsed["error"]["type"], "RuntimeError")

    def test_fold_metrics_include_per_state_aggregate(self) -> None:
        fold = lambda energy, state_value: {"metrics": {
            "energy_mae_ev": energy, "force_mae_ev_per_ang": energy * 2,
            "energy_mae_by_state_ev": {"S0": state_value},
            "force_mae_by_state_ev_per_ang": {"S0": state_value * 2},
        }}
        metrics = evaluation.aggregate_fold_metrics({"fold_1": fold(1.0, 3.0), "fold_2": fold(3.0, 5.0)})
        self.assertEqual(metrics["energy_mae_ev"], {"mean": 2.0, "variance": 1.0})
        self.assertEqual(metrics["energy_mae_by_state_ev"]["S0"], {"mean": 4.0, "variance": 1.0})


class UnifiedTrainTests(unittest.TestCase):
    def _run_dispatch(self, mode: str) -> tuple[MagicMock, MagicMock, MagicMock, str]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "run.json"
            config_path.write_text("{}", encoding="utf-8")
            run_dir = root / "run_0"
            run_dir.mkdir()
            config = {"mode": mode, "device": "cpu", "verbose": False}
            run_state = MagicMock()
            run_state.complete.return_value = run_dir / "result.json"
            stderr = io.StringIO()
            with (
                patch.object(train, "load_config", return_value=(config, ["visible warning"])),
                patch.object(train.state, "RunState", return_value=run_state),
                patch.object(train.model, "validate_device", return_value="cpu"),
                patch.object(train, "_run_scratch_stage") as scratch,
                patch.object(train, "_run_transfer_stage") as transfer,
                redirect_stderr(stderr),
            ):
                result = train.run_config(config_path, root, run_dir)
            self.assertEqual(result, run_dir / "result.json")
            return scratch, transfer, run_state, stderr.getvalue()

    def test_both_dispatches_independent_lf_and_hf_stages(self) -> None:
        scratch, transfer, run_state, stderr = self._run_dispatch("both")
        self.assertEqual([call.args[0] for call in scratch.call_args_list], ["lf", "hf"])
        transfer.assert_not_called()
        run_state.complete.assert_called_once()
        self.assertIn("Warning: visible warning", stderr)

    def test_transfer_dispatches_only_transfer_stage(self) -> None:
        scratch, transfer, _, _ = self._run_dispatch("transfer")
        scratch.assert_not_called()
        transfer.assert_called_once()


class JobControllerTests(unittest.TestCase):
    def test_scheduler_uses_strict_config_and_unified_train(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "job.json"
            config_path.write_text(json.dumps(_lf_config(device="cpu")), encoding="utf-8")
            self.assertEqual(job_controller._read_job_options(config_path), (False, "cpu"))
            run_dir = state.reserve_run_dir(root)
            job = job_controller.Job(config_path, run_dir)
            with patch("scripts.train.run_config", return_value=run_dir / "result.json") as run_config:
                self.assertEqual(job_controller._run_job(job), run_dir / "result.json")
            run_config.assert_called_once_with(config_path, run_dir.parent, run_dir)

    def test_gpu_worker_and_selection_validation(self) -> None:
        self.assertEqual(job_controller._parse_gpu_indices("2,0", 3), [2, 0])
        with self.assertRaises(ValueError):
            job_controller._parse_gpu_indices("0,0", 3)
        jobs: queue.Queue[object] = queue.Queue()
        results: queue.Queue[object] = queue.Queue()
        job = job_controller.Job(Path("job.json"), Path("run_0"))
        jobs.put(job)
        jobs.put(None)
        with patch.object(job_controller, "_run_job", return_value=Path("run_0/result.json")):
            job_controller._gpu_worker(1, jobs, results)
        self.assertEqual(results.get_nowait()[1], 1)

