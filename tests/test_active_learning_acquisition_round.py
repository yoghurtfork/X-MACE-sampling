"""Exercise acquisition through round orchestration and persisted state."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ACTIVE_LEARNING_DIR = Path(__file__).resolve().parent.parent / 'scripts' / 'active-learning'
if str(ACTIVE_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVE_LEARNING_DIR))

import run as active_learning_run
from state import StateStore, new_state


class AcquisitionRoundTests(unittest.TestCase):
    def test_rounds_persist_selection_and_use_enlarged_training_pool(self):
        for fraction in (0.5, 0):
            with self.subTest(fraction=fraction), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state = new_state(
                    input_file=root / 'config.json', run_directory=root, config={},
                    resume_identity={}, grid_size=10, grid_shape=(2, 5),
                    energy_key='energy', forces_key='forces', initial_acquired_indices=[0],
                )
                store = StateStore(root / 'result.json')
                grid = Mock(size=10)
                grid.reveal.side_effect = lambda indices: list(indices)
                grid.unacquired_indices.side_effect = lambda indices: np.array(
                    sorted(set(range(10)) - set(indices)), dtype=int
                )
                grid.prediction_atoms.side_effect = lambda indices: list(indices)
                builder = Mock()
                builder.load.side_effect = lambda atoms, **kwargs: atoms
                config = Mock(
                    n_rounds=2, seed=42, acquired_per_round=2,
                    acquire_from_uncertain_fraction=fraction,
                    trainer_options={'verbose': False},
                )
                committee = SimpleNamespace(training={'E0s': {}}, model_paths={'0': 'model.pt'})

                def predict(models, loader, **kwargs):
                    return SimpleNamespace(energies=np.array(loader, dtype=float), forces=None)

                def uncertainty(energies, forces, **kwargs):
                    return SimpleNamespace(score=energies, energy_std=energies,
                                           force_std=np.zeros(len(energies)))

                with (
                    patch.object(active_learning_run, 'train_round_committee', return_value=committee) as train,
                    patch.object(active_learning_run, 'load_model'),
                    patch.object(active_learning_run, 'predict_committee', side_effect=predict),
                    patch.object(active_learning_run, 'committee_uncertainty', side_effect=uncertainty),
                    patch('builtins.print'),
                ):
                    active_learning_run._run_rounds(
                        state=state, store=store, config=config, grid=grid,
                        test_atoms=None, data_builder_class=Mock(return_value=builder),
                        trainer_class=Mock(), tester_class=Mock(), loss_fn=Mock(), device='cpu',
                    )
                saved = store.load()
                self.assertEqual(saved['acquired_indices'], state['acquired_indices'])
                self.assertEqual(saved['final_production_model']['status'], 'pending')
                if fraction == 0:
                    self.assertEqual(train.call_count, 1)
                    self.assertEqual(saved['acquired_indices'], [0])
                    self.assertIn('termination_reason', saved)
                    self.assertEqual(saved['rounds'][0]['selection'],
                                     {'eligible_indices': [], 'acquired_indices': []})
                    continue
                acquired = {0}
                for number, record in enumerate(saved['rounds']):
                    self.assertEqual(train.call_args_list[number].kwargs['acquired_atoms'], sorted(acquired))
                    remaining = sorted(set(range(10)) - acquired, reverse=True)
                    eligible = remaining[:int(np.ceil(fraction * len(remaining)))]
                    expected = sorted(np.random.default_rng(42 + number).choice(eligible, 2, replace=False).tolist())
                    self.assertEqual(record['selection'],
                                     {'eligible_indices': eligible, 'acquired_indices': expected})
                    acquired.update(expected)
                self.assertEqual(saved['acquired_indices'], sorted(acquired))
                self.assertEqual(len(acquired), 5)
