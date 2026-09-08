"""Regression tests for fraction-based active-learning acquisition."""

import sys
import unittest
from pathlib import Path

import numpy as np

ACTIVE_LEARNING_DIR = Path(__file__).resolve().parent.parent / "scripts" / "active-learning"
if str(ACTIVE_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVE_LEARNING_DIR))

from selection import select_acquisitions


class AcquisitionTests(unittest.TestCase):
    def select(self, scores, **overrides):
        kwargs = dict(already_acquired=[], acquired_per_round=2,
                      acquire_from_uncertain_fraction=0.5,
                      rng=np.random.default_rng(42))
        kwargs.update(overrides)
        return select_acquisitions(scores, **kwargs)

    def test_ranking_rounding_ties_and_exclusion(self):
        result = self.select([100, 2, 7, 7, 1, 4], already_acquired=[0])
        self.assertEqual(result.eligible_indices.tolist(), [2, 3, 5])
        self.assertEqual(len(result.acquired_indices), 2)
        self.assertEqual(len(set(result.acquired_indices)), 2)
        self.assertTrue(set(result.acquired_indices).issubset({2, 3, 5}))

    def test_ties_at_cutoff_use_index(self):
        result = self.select([3, 3, 3, 3])
        self.assertEqual(result.eligible_indices.tolist(), [0, 1])

    def test_small_pool_acquires_all_candidates(self):
        result = self.select([1, 4, 2], acquired_per_round=10,
                             acquire_from_uncertain_fraction=0.1)
        self.assertEqual(result.acquired_indices.tolist(), [1])

    def test_fraction_endpoints_and_exhaustion(self):
        zero = self.select([1, 2], acquire_from_uncertain_fraction=0)
        self.assertEqual(zero.eligible_indices.tolist(), [])
        self.assertEqual(zero.acquired_indices.tolist(), [])
        full = self.select([0, 0, 0], acquire_from_uncertain_fraction=1,
                           acquired_per_round=10, already_acquired=[1])
        self.assertEqual(full.acquired_indices.tolist(), [0, 2])
        empty = self.select([1, 2], already_acquired=[0, 1])
        self.assertEqual(empty.acquired_indices.tolist(), [])

    def test_seeded_sampling_is_reproducible_and_varies(self):
        first = self.select(np.arange(100))
        repeat = self.select(np.arange(100))
        other = self.select(np.arange(100), rng=np.random.default_rng(43))
        np.testing.assert_array_equal(first.acquired_indices, repeat.acquired_indices)
        self.assertFalse(np.array_equal(first.acquired_indices, other.acquired_indices))

    def test_invalid_inputs(self):
        cases = [dict(acquired_per_round=x) for x in (0, -1, True, 1.5)]
        cases += [dict(acquire_from_uncertain_fraction=x)
                  for x in (-0.1, 1.1, True, '0.5', float('nan'), float('inf'))]
        cases += [dict(already_acquired=x) for x in ([0, 0], [-1], [3], [True])]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.select([1, 2, 3], **kwargs)
        for scores in ([], [[1]], [float('nan')], [float('inf')]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                self.select(scores)
