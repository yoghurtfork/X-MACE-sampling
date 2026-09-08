"""Final production checkpoints exercise real serialization and evaluation."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ACTIVE_LEARNING_DIR = Path(__file__).resolve().parent.parent / 'scripts' / 'active-learning'
if str(ACTIVE_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVE_LEARNING_DIR))

import run as active_learning_run
from state import StateStore, new_state


class WeightTester:
    """Expose the loaded epoch's weight as a metric through the real evaluator."""

    def run_test(self, model, loader):
        self.weight = model.weight.item()

    def get_energy_mae(self):
        return self.weight

    def get_force_mae(self):
        return self.weight * 2

    def get_energy_mae_by_state(self):
        return [self.weight]

    def get_force_mae_by_state(self):
        return [self.weight * 2]


class FinalCheckpointTests(unittest.TestCase):
    def run_training(self, root, interval, *, evaluate=False, epochs=5):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        config = SimpleNamespace(
            seed=42, r_max=5.0, energy_key='energy', forces_key='forces',
            e0s={}, batch_size=2, lf_checkpoint=root / 'lf.pt', strategy='naive',
            strategy_kwargs={}, final_max_epochs=epochs, final_learning_rate=0.01,
            trainer_options={'verbose': False},
            final_production_model_checkpoint_epochs=interval,
        )
        state = new_state(
            input_file=root / 'config.json', run_directory=root, config={},
            resume_identity={}, grid_size=2, grid_shape=(1, 2),
            energy_key='energy', forces_key='forces', initial_acquired_indices=[0, 1],
        )
        store = StateStore(root / 'result.json')
        builder = Mock()
        trainer = Mock(optimiser_weight_decay=0.0)

        def train_epoch(model, *args, **kwargs):
            with torch.no_grad():
                model.weight.add_(1)
            return {'loss': 1.0, 'energy_mae': 2.0, 'force_mae': 3.0}

        trainer._run_epoch.side_effect = train_epoch
        with (
            patch.object(active_learning_run, 'load_model', return_value=model),
            patch.object(active_learning_run, 'resolved_e0s', return_value={}),
            patch.object(active_learning_run, '_seed_everything'),
            patch.object(active_learning_run, 'apply_training_strategy', side_effect=lambda m, _: m),
            patch.object(active_learning_run, 'evaluate_checkpoint_models',
                         wraps=active_learning_run.evaluate_checkpoint_models) as evaluator,
            patch('mace.training.trainer.build_optimiser',
                  return_value=SimpleNamespace(param_groups=[{'lr': 0.01}])),
            patch('builtins.print'),
        ):
            active_learning_run._train_final_production_model(
                state=state, store=store, config=config,
                grid=Mock(reveal=Mock(return_value=[0, 1])),
                test_atoms=[2] if evaluate else None,
                data_builder_class=Mock(return_value=builder),
                trainer_class=Mock(return_value=trainer),
                tester_class=lambda **kwargs: WeightTester(),
                loss_fn=Mock(), device=torch.device('cpu'), run_dir=root,
            )
        self.assertEqual(evaluator.call_count, int(evaluate))
        self.assertEqual(model.weight.item(), epochs)
        saved = store.load()
        self.assertEqual(saved['status'], 'completed')
        return saved['final_production_model']

    def test_checkpoint_timing_and_model_files_without_test_data(self):
        for interval, epochs, expected in (
            (2, 5, [2, 4]), (1, 3, [1, 2, 3]), (2, 4, [2, 4]),
            (10, 5, []), (None, 5, []),
        ):
            with self.subTest(interval=interval, epochs=epochs), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.run_training(root, interval, epochs=epochs)
                model_dir = root / 'final_production_model'
                final_path = model_dir / 'final_production_model.pt'
                self.assertEqual(result['model_path'], str(final_path))
                self.assertEqual(torch.load(final_path, weights_only=False).weight.item(), epochs)
                self.assertEqual(
                    {p.name for p in model_dir.iterdir()},
                    {'final_production_model.pt'} | {f'checkpoint_epoch_{e}.pt' for e in expected},
                )
                entries = result['history']['checkpoint_models']
                self.assertEqual([entry['epoch'] for entry in entries], expected)
                for entry in entries:
                    path = model_dir / f"checkpoint_epoch_{entry['epoch']}.pt"
                    self.assertEqual(entry, {'epoch': entry['epoch'], 'path': str(path)})
                    self.assertEqual(torch.load(path, weights_only=True)['weight'].item(), entry['epoch'])
                self.assertNotIn('hf_test_metrics', result)

    def test_checkpoint_evaluation_records_each_epochs_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_training(Path(directory), 2, evaluate=True)
            entries = result['history']['checkpoint_models']
            self.assertEqual([entry['epoch'] for entry in entries], [2, 4])
            for entry in entries:
                self.assertTrue(Path(entry['model_path']).is_file())
                self.assertEqual(entry['test_energy_mae'], entry['epoch'])
                self.assertEqual(entry['test_force_mae'], 2 * entry['epoch'])
                self.assertIn('test_1', entry['test_metrics'])
            self.assertEqual(result['hf_test_metrics']['energy_mae_ev'], 5)

    def test_disabled_checkpoints_still_evaluate_final_model(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_training(Path(directory), None, evaluate=True)
            self.assertEqual(result['history']['checkpoint_models'], [])
            self.assertEqual(result['hf_test_metrics']['energy_mae_ev'], 5)
