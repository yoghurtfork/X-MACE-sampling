"""Validation of active-learning acquisition configuration."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ACTIVE_LEARNING_DIR = Path(__file__).resolve().parent.parent / "scripts" / "active-learning"
if str(ACTIVE_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVE_LEARNING_DIR))

from config import load_config


class AcquisitionConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'checkpoint.pt').touch()
        (self.root / 'data.xyz').touch()
        self.config = dict(lf_checkpoint='checkpoint.pt', hf_xyz='data.xyz',
                           grid_shape=[2, 3], initial_acquired_count=2,
                           n_rounds=3, k=2, acquired_per_round=4,
                           acquire_from_uncertain_fraction=0.1)

    def load(self, config):
        path = self.root / 'config.json'
        path.write_text(json.dumps(config))
        return load_config(path)

    def test_valid_values_and_endpoints(self):
        for fraction in (0, 0.1, 1):
            with self.subTest(fraction=fraction):
                loaded = self.load({**self.config, 'acquire_from_uncertain_fraction': fraction})
                self.assertEqual(loaded.acquired_per_round, 4)
                self.assertEqual(loaded.acquire_from_uncertain_fraction, fraction)

    def test_missing_fields(self):
        for field in ('acquired_per_round', 'acquire_from_uncertain_fraction'):
            config = self.config.copy()
            del config[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.load(config)

    def test_invalid_values(self):
        values = {'acquired_per_round': [0, -1, True, 1.5, '4', None],
                  'acquire_from_uncertain_fraction': [-0.1, 1.1, True, '0.5', None,
                                                     float('nan'), float('inf'), -float('inf')]}
        for field, invalid in values.items():
            for value in invalid:
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    self.load({**self.config, field: value})

    def test_legacy_fields_rejected(self):
        for field in ('max_seeds_per_round', 'uncertainty_threshold'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Unknown'):
                self.load({**self.config, field: 1})

    def test_final_checkpoint_interval(self):
        field = 'final_production_model_checkpoint_epochs'
        self.assertIsNone(getattr(self.load(self.config), field))
        for value in (None, 1, 10):
            with self.subTest(value=value):
                config = self.load({**self.config, field: value, 'checkpoint_epochs': 3})
                self.assertEqual(getattr(config, field), value)
                self.assertEqual(config.checkpoint_epochs, 3)
        for value in (0, -1, True, 1.5, '10', float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, field):
                self.load({**self.config, field: value})
