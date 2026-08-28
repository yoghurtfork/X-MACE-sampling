"""Regression tests for multi-GPU job scheduling helpers."""

from __future__ import annotations

import json
import os
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import torch

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
                        "checkpoint_epochs": 3,
                        "verbose": True,
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
                patch.object(single_trainer, "_save_loss_plot", return_value="loss.png") as save_loss_plot,
                patch.object(single_trainer, "_save_epoch_mae_plot", return_value="mae.png") as save_mae_plot,
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
            self.assertEqual(kwargs["checkpoint_epochs"], 3)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["run"], "just_full")
            self.assertIn("full_high_fidelity_model", result["models"])
            self.assertNotIn("base_model", result["models"])
            self.assertNotIn("final_metrics_comparison_plot", result["artifacts"])
            save_loss_plot.assert_called_once()
            save_mae_plot.assert_called_once()

    def test_single_stage_verbose_false_skips_plots_and_artifacts(self) -> None:
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
                        "cross_validation": False,
                        "verbose": False,
                    }
                ),
                encoding="utf-8",
            )
            model_run = {
                "model": {"weights": "test"},
                "history": {"best_epoch": 1},
                "metrics": {"energy": 0.1},
                "training_seconds": 1.0,
                "max_epochs": 1,
                "learning_rate": 0.001,
                "E0s": {"C": -1.5},
            }
            with (
                patch.object(single_trainer, "_import_project_modules", return_value=self._modules()),
                patch.object(single_trainer, "_read_atoms", side_effect=[[1, 2], [3]]),
                patch.object(single_trainer, "_validate_device", return_value="cpu"),
                patch.object(single_trainer, "_train_model", return_value=model_run),
                patch.object(single_trainer, "_save_loss_plot") as save_loss_plot,
                patch.object(single_trainer, "_save_epoch_mae_plot") as save_mae_plot,
            ):
                result_path = single_trainer.run_config(config_path, root, run_dir)

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed")
            self.assertIn("base_model", result["models"])
            self.assertNotIn("artifacts", result)
            save_loss_plot.assert_not_called()
            save_mae_plot.assert_not_called()

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
                        "checkpoint_epochs": 4,
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
            self.assertEqual(kwargs["checkpoint_epochs"], 4)
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

    class _Model(torch.nn.Module):
        pass

    class _Trainer:
        def __init__(self, **kwargs):
            pass

        def train_model(
            self, model, train_loader, valid_loader, loss_fn,
            checkpoint_epoch=None,
        ):
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
            patch.object(helper, "_save_model"),
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

    def test_cross_validation_verbose_false_skips_fold_plots(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_dir,
            patch.object(helper, "seed_everything"),
            patch.object(helper, "_evaluate", return_value=self._metrics()),
            patch.object(helper, "_save_loss_plot") as save_loss_plot,
            patch.object(helper, "_save_epoch_mae_plot") as save_mae_plot,
            patch.object(helper, "_save_model"),
            patch("builtins.print") as output,
        ):
            result = helper._train_k_fold_models(
                initial_model=self._Model(),
                all_atoms=["train_0", "train_1", "train_2", "train_3"],
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
                generate_plots=False,
            )

        self.assertNotIn("artifacts", result)
        save_loss_plot.assert_not_called()
        save_mae_plot.assert_not_called()
        output.assert_not_called()

    def test_cross_validation_uses_and_records_distinct_fold_seeds(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_dir,
            patch.object(helper, "seed_everything") as seed_everything,
            patch.object(helper, "_evaluate", return_value=self._metrics()),
            patch.object(helper, "_save_model"),
        ):
            result = helper._train_k_fold_models(
                initial_model=self._Model(),
                all_atoms=["train_0", "train_1", "train_2", "train_3"],
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
                generate_plots=False,
            )

        self.assertEqual(seed_everything.call_args_list, [call(43), call(44)])
        self.assertEqual(result["folds"]["model_1"]["fold_seed"], 43)
        self.assertEqual(result["folds"]["model_2"]["fold_seed"], 44)

    def test_cross_validation_progress_label_reports_each_fold(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_dir,
            patch.object(helper, "seed_everything"),
            patch.object(helper, "_evaluate", return_value=self._metrics()),
            patch.object(helper, "_save_model"),
            patch("builtins.print") as output,
        ):
            helper._train_k_fold_models(
                initial_model=self._Model(),
                all_atoms=["train_0", "train_1", "train_2", "train_3"],
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
                generate_plots=False,
                progress_label="[active-learning] round 1",
            )

        self.assertEqual(
            output.call_args_list,
            [
                call(
                    "[active-learning] round 1 | starting fold 1/2 "
                    "(train=2, validation=2, seed=43)",
                    flush=True,
                ),
                call(
                    "[active-learning] round 1 | starting fold 2/2 "
                    "(train=2, validation=2, seed=44)",
                    flush=True,
                ),
            ],
        )


class TrainingCheckpointTests(unittest.TestCase):
    class _Builder(AutomaticE0Tests._Builder):
        pass

    class _Model(AutomaticE0Tests._Model):
        pass

    class _InterruptingTrainer:
        def __init__(self, **kwargs):
            pass

        def train_model(
            self, model, train_loader, valid_loader, loss_fn,
            checkpoint_epoch=None,
        ):
            raise KeyboardInterrupt

    class _CompletedTrainer:
        checkpoint_epoch = None

        def __init__(self, **kwargs):
            pass

        def train_model(
            self, model, train_loader, valid_loader, loss_fn,
            checkpoint_epoch=None,
        ):
            self.__class__.checkpoint_epoch = checkpoint_epoch
            return model, {"best_epoch": 1}

    def _kwargs(self, run_dir: Path) -> dict[str, object]:
        return {
            "initial_model": self._Model(),
            "all_atoms": ["train_0", "train_1"],
            "test_atoms": ["test_0"],
            "model_prefix": "model",
            "run_dir": run_dir,
            "data_builder_class": self._Builder,
            "tester": object(),
            "loss_fn": object(),
            "device": "cpu",
            "seed": 42,
            "k": 2,
            "r_max": 5.0,
            "batch_size": 2,
            "max_epochs": 1,
            "learning_rate": 0.001,
            "trainer_options": {},
            "energy_key": "REF_energy",
            "forces_key": "REF_forces",
            "e0s": None,
        }

    def test_interruption_saves_current_fold_and_reports_it(self) -> None:
        snapshots = []
        with tempfile.TemporaryDirectory() as temporary_dir, patch.object(
            helper, "_save_model", side_effect=lambda model, path: path
        ) as save_model:
            with self.assertRaises(KeyboardInterrupt):
                helper._train_k_fold_models(
                    trainer_class=self._InterruptingTrainer,
                    on_checkpoint=snapshots.append,
                    **self._kwargs(Path(temporary_dir)),
                )

        self.assertTrue(save_model.called)
        self.assertEqual(snapshots[-1]["current_fold"]["status"], "interrupted")
        self.assertEqual(snapshots[-1]["completed_folds"], 0)

    def test_completed_fold_is_saved_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir, patch.object(
            helper, "_save_model", side_effect=lambda model, path: path
        ) as save_model, patch.object(
            helper, "_evaluate", side_effect=RuntimeError("evaluation failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                helper._train_k_fold_models(
                    trainer_class=self._CompletedTrainer,
                    checkpoint_epochs=2,
                    **self._kwargs(Path(temporary_dir)),
                )

        self.assertTrue(save_model.called)
        self.assertEqual(self._CompletedTrainer.checkpoint_epoch, 2)

    def test_checkpoint_history_contains_test_metrics(self) -> None:
        model = torch.nn.Linear(1, 1)
        checkpoint_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        history = {"checkpoint_models": [checkpoint_state]}
        checkpoint_metrics = {
            "energy_mae_ev": 0.2,
            "force_mae_ev_per_ang": 0.3,
            "energy_mae_by_state_ev": {"S0": 0.2},
            "force_mae_by_state_ev_per_ang": {"S0": 0.3},
        }
        with tempfile.TemporaryDirectory() as temporary_dir, patch.object(
            helper, "_evaluate", return_value=checkpoint_metrics
        ):
            checkpoint_path = Path(temporary_dir) / "model.pt"
            helper._evaluate_checkpoint_models(
                model,
                history,
                checkpoint_epochs=5,
                model_path=checkpoint_path,
                test_loader=object(),
                tester=object(),
                device=torch.device("cpu"),
            )
            saved_path = checkpoint_path.with_name(
                "model_checkpoint_epoch_5.pt"
            )
            self.assertTrue(saved_path.is_file())
            self.assertIsInstance(
                helper._load_model(saved_path, torch.device("cpu")), torch.nn.Linear
            )

        self.assertEqual(
            history["checkpoint_models"],
            [{
                "epoch": 5,
                "model_path": str(saved_path),
                "test_energy_mae": 0.2,
                "test_force_mae": 0.3,
            }],
        )
        json.dumps(history)

    def test_checkpoint_epochs_validation(self) -> None:
        self.assertIsNone(helper._checkpoint_epochs_from_config({}))
        self.assertIsNone(helper._checkpoint_epochs_from_config({"checkpoint_epochs": None}))
        self.assertEqual(
            helper._checkpoint_epochs_from_config({"checkpoint_epochs": 5}), 5
        )
        for value in (0, -1, True, 1.5, "5", []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "checkpoint_epochs"):
                    helper._checkpoint_epochs_from_config({"checkpoint_epochs": value})


class TrainingStrategyTests(unittest.TestCase):
    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node_embedding = torch.nn.Linear(1, 1)
            self.interactions = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
            self.readouts = torch.nn.Linear(1, 1)

    def test_strategy_defaults_to_naive_and_rejects_invalid_values(self) -> None:
        self.assertEqual(helper._strategy_from_config({}), "naive")
        self.assertEqual(helper._strategy_kwargs_from_config({}, "naive"), {})
        with self.assertRaisesRegex(ValueError, "JSON string"):
            helper._strategy_from_config({"strategy": False})
        with self.assertRaisesRegex(ValueError, "naive.*freeze"):
            helper._strategy_from_config({"strategy": "invalid"})
        with self.assertRaisesRegex(ValueError, "not supported"):
            helper._strategy_kwargs_from_config(
                {"strategy_kwargs": {"unexpected": True}}, "naive"
            )

    def test_freeze_strategy_freezes_requested_layers_only(self) -> None:
        model = self._Model()
        transformed = helper._apply_training_strategy(model, "freeze")

        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in transformed.node_embedding.parameters()
        ))
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in transformed.interactions.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in transformed.readouts.parameters()
        ))

    def test_freeze_strategy_accepts_frozen_layers_from_json(self) -> None:
        strategy_kwargs = helper._strategy_kwargs_from_config(
            {"strategy_kwargs": {"frozen_layers": ["node_embedding"]}},
            "freeze",
        )
        transformed = helper._apply_training_strategy(
            self._Model(), "freeze", strategy_kwargs
        )

        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in transformed.node_embedding.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in transformed.interactions.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
