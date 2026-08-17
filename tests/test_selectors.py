"""Tests for sampling-method selectors."""

import unittest

import numpy as np

from sampling_methods.selectors import get_selector, stratified_sampling


class StratifiedSamplingTests(unittest.TestCase):
    def setUp(self):
        # The second state sets the ranking, ensuring multi-state max reduction.
        scores = np.arange(12, dtype=float)
        self.descriptor_matrix = np.column_stack(
            [np.zeros_like(scores), scores, np.ones_like(scores)]
        )
        self.kwargs = {
            "top_fraction": 0.25,
            "middle_fraction": 0.25,
            "bottom_fraction": 0.5,
            "top_n": 2,
            "middle_n": 2,
            "bottom_n": 3,
            "random_state": 7,
        }

    def test_samples_only_from_their_declared_strata(self):
        selected = get_selector(
            "stratified_sampling", self.descriptor_matrix, 7, **self.kwargs
        )

        top_pool = {9, 10, 11}
        middle_pool = {6, 7, 8}
        bottom_pool = {0, 1, 2, 3, 4, 5}
        self.assertEqual(len(selected), 7)
        self.assertEqual(len(np.unique(selected)), 7)
        self.assertEqual(sum(index in top_pool for index in selected), 2)
        self.assertEqual(sum(index in middle_pool for index in selected), 2)
        self.assertEqual(sum(index in bottom_pool for index in selected), 3)

    def test_sampling_is_reproducible_and_uses_maximum_state_score(self):
        first = stratified_sampling(self.descriptor_matrix, 7, **self.kwargs)
        second = stratified_sampling(self.descriptor_matrix, 7, **self.kwargs)
        np.testing.assert_array_equal(first, second)

        scores = np.max(self.descriptor_matrix, axis=1)
        self.assertSetEqual(set(np.argsort(-scores, kind="stable")[:3]), {9, 10, 11})

    def test_invalid_stratified_sampling_arguments_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            stratified_sampling(
                self.descriptor_matrix, 7, **{**self.kwargs, "bottom_fraction": 0.4}
            )
        with self.assertRaisesRegex(ValueError, "must equal"):
            stratified_sampling(self.descriptor_matrix, 6, **self.kwargs)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            stratified_sampling(
                self.descriptor_matrix, 9, **{**self.kwargs, "top_n": 4}
            )
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            stratified_sampling(np.arange(12), 7, **self.kwargs)


if __name__ == "__main__":
    unittest.main()
