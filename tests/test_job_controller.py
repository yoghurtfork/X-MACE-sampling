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
from scripts import single_trainer
from scripts import helper


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

    def test_read_job_options_defaults_and_validates_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config_path = Path(temporary_dir) / "job.json"
            config_path.write_text(
                json.dumps({"transfer_learning": False}), encoding="utf-8"
            )
            self.assertEqual(
                job_controller._read_job_options(config_path),
                (False, False, "cpu", "both"),
            )
            config_path.write_text(
                json.dumps({"run": "not-a-stage"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "'run'"):
                job_controller._read_job_options(config_path)

    def test_run_job_routes_single_stage_modes_to_single_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run_0"
            run_dir.mkdir()
            job = job_controller.Job(
                Path("job.json"), False, run_dir, "just_full"
            )
            with patch("scripts.single_trainer.run_config", return_value=run_dir / "result.json") as run_config:
                self.assertEqual(job_controller._run_job(job), run_dir / "result.json")
            run_config.assert_called_once_with(Path("job.json"), run_dir.parent, run_dir)


class SingleTrainerTests(unittest.TestCase):
    class _Metadata:
        atomic_numbers = [6]
        atomic_energies = [-1.5]

    class _Loss:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to(self, device):
            return self

    class _Builder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self, atoms, **kwargs):
            return atoms

        def get_metadata(self):
            return SingleTrainerTests._Metadata()

    class _Model:
        def to(self, device):
            return self

    def _modules(self):
        return (
            type("Modules", (), {"InvariantsWeightedEnergyForcesNacsDipoleLoss": self._Loss}),
            self._Builder,
            lambda **kwargs: object(),
            object,
            object,
            object,
            lambda metadata, **kwargs: self._Model(),
            object,
            object,
        )

    def test_single_stage_uses_selected_inputs_and_shared_options(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config_path = root / "single.json"
            run_dir = root / "run_0"
            run_dir.mkdir()
            config_path.write_text(
                json.dumps(
                    {
                        "transfer_learning": False,
                        "run": "just_full",
                        "transfer_xyz": "full.xyz",
                        "transfer_test_xyz": "full_test.xyz",
                        "cross_validation": False,
                        "seed": 7,
                        "batch_size": 3,
                        "max_epochs": 12,
                        "full_learning_rate": 0.02,
                        "full_E0s": {"C": -1.5},
                        "preset": "test-preset",
                        "load_base": None,
                    }
                ),
                encoding="utf-8",
            )
            model_run = {
                "model": {"weights": "test"},
                "history": {"best_epoch": 4},
                "metrics": {"energy": 0.1},
                "training_seconds": 1.0,
                "max_epochs": 12,
                "learning_rate": 0.02,
                "E0s": {"C": -1.5},
            }
            with (
                patch.object(single_trainer, "_import_project_modules", return_value=self._modules()),
                patch.object(single_trainer, "_read_atoms", side_effect=[[1, 2, 3, 4], [5]]),
                patch.object(single_trainer, "_validate_device", return_value="cpu"),
                patch.object(single_trainer, "_train_model", return_value=model_run) as train_model,
                patch.object(single_trainer, "_save_loss_plot", return_value="loss.png"),
                patch.object(single_trainer, "_save_epoch_mae_plot", return_value="mae.png"),
            ):
                result_path = single_trainer.run_config(config_path, root, run_dir)
            kwargs = train_model.call_args.kwargs
            self.assertEqual(kwargs["max_epochs"], 12)
            self.assertEqual(kwargs["learning_rate"], 0.02)
            self.assertEqual(kwargs["batch_size"], 3)
            self.assertEqual(kwargs["seed"], 7)
            self.assertEqual(kwargs["preset"], "test-preset")
            self.assertEqual(kwargs["load_base"], None)
            self.assertEqual(kwargs["e0s"]["C"], -1.5)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["run"], "just_full")
            self.assertIn("full_high_fidelity_model", result["models"])
            self.assertNotIn("base_model", result["models"])
            self.assertNotIn("final_metrics_comparison_plot", result["artifacts"])

    def test_base_cross_validation_uses_base_inputs_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config_path = root / "single.json"
            run_dir = root / "run_0"
            run_dir.mkdir()
            config_path.write_text(
                json.dumps(
                    {
                        "transfer_learning": False,
                        "run": "just_base",
                        "base_xyz": "base.xyz",
                        "base_test_xyz": "base_test.xyz",
                        "cross_validation": True,
                        "k": 2,
                        "base_max_epochs": 9,
                        "base_learning_rate": 0.03,
                    }
                ),
                encoding="utf-8",
            )
            cv_run = {
                "aggregate_test_metrics": {"energy": 0.2},
                "model_paths": {"model_1": "one.pt", "model_2": "two.pt"},
                "artifacts": {"model_1": {}, "model_2": {}},
                "completed_folds": 2,
                "total_folds": 2,
            }
            with (
                patch.object(single_trainer, "_import_project_modules", return_value=self._modules()),
                patch.object(single_trainer, "_read_atoms", side_effect=[[1, 2, 3, 4], [5]]),
                patch.object(single_trainer, "_validate_device", return_value="cpu"),
                patch.object(single_trainer, "_train_k_fold_models", return_value=cv_run) as train_models,
                patch.object(single_trainer, "_save_fold_selection_plot", return_value="selection.png"),
            ):
                result_path = single_trainer.run_config(config_path, root, run_dir)
            kwargs = train_models.call_args.kwargs
            self.assertEqual(kwargs["k"], 2)
            self.assertEqual(kwargs["max_epochs"], 9)
            self.assertEqual(kwargs["learning_rate"], 0.03)
            self.assertIsNone(kwargs["e0s"])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["E0s"], {"base": {"C": -1.5}})
            self.assertIn("base_models", result["models"])
            self.assertNotIn("full_high_fidelity_models", result["models"])


class AutomaticE0Tests(unittest.TestCase):
    class _Builder:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.load_calls = []
            self.__class__.instances.append(self)

        def load(self, atoms, **kwargs):
            self.load_calls.append((list(atoms), kwargs))
            return atoms

        def get_metadata(self):
            return type(
                "Metadata", (), {"atomic_numbers": [1], "atomic_energies": [-0.5]}
            )()

    class _Model:
        def to(self, device):
            return self

        def cpu(self):
            return self

    class _Trainer:
        def __init__(self, **kwargs):
            pass

        def train_model(self, model, train_loader, valid_loader, loss_fn):
            return model, {"best_epoch": 1}

    @staticmethod
    def _metrics():
        states = {f"S{state}": 0.1 for state in range(3)}
        return {
            "energy_mae_ev": 0.1,
            "force_mae_ev_per_ang": 0.1,
            "energy_mae_by_state_ev": states,
            "force_mae_by_state_ev_per_ang": states,
        }

    def test_cross_validation_fits_automatic_e0s_from_training_pool(self) -> None:
        self._Builder.instances.clear()
        with (
            tempfile.TemporaryDirectory() as temporary_dir,
            patch.object(helper, "seed_everything"),
            patch.object(helper, "_evaluate", return_value=self._metrics()),
            patch.object(helper, "_save_loss_plot", return_value="loss.png"),
            patch.object(helper, "_save_epoch_mae_plot", return_value="mae.png"),
            patch.object(helper.torch, "save"),
        ):
            training_pool = ["train_0", "train_1", "train_2", "train_3"]
            helper._train_k_fold_models(
                initial_model=self._Model(),
                all_atoms=training_pool,
                test_atoms=["test_0"],
                model_prefix="model",
                run_dir=Path(temporary_dir),
                data_builder_class=self._Builder,
                trainer_class=self._Trainer,
                tester=object(),
                loss_fn=object(),
                device="cpu",
                seed=42,
                k=2,
                r_max=5.0,
                batch_size=2,
                max_epochs=1,
                learning_rate=0.001,
                trainer_options={},
                energy_key="REF_energy",
                forces_key="REF_forces",
                e0s=None,
            )
        builder = self._Builder.instances[0]
        self.assertIsNone(builder.kwargs["E0s"])
        self.assertEqual(builder.load_calls[0][0], training_pool)
        self.assertEqual(builder.load_calls[1][0], ["test_0"])


if __name__ == "__main__":
    unittest.main()
