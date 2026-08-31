"""Tests for active-learning result-state storage."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_LEARNING_DIR = PROJECT_ROOT / "scripts" / "active-learning"
if str(ACTIVE_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVE_LEARNING_DIR))

import state


class ActiveLearningStateTests(unittest.TestCase):
    def test_completed_round_records_per_point_uncertainty(self) -> None:
        run_state = state.new_state(
            input_file=Path("input.json"), run_directory=Path("run"), config={},
            resume_identity={}, grid_size=2, grid_shape=(1, 2),
            energy_key="energy", forces_key="forces", initial_acquired_indices=[0],
        )
        state.start_round(run_state, round_number=0, acquired_before=[0])
        state.complete_current_round(
            run_state,
            committee={},
            uncertainty={},
            committee_uncertainty_by_index=[
                {"index": 1, "energy_std": 0.02, "force_std": 0.04, "score": 0.06}
            ],
            selection={},
            acquired_after=[0],
        )

        self.assertEqual(
            run_state["rounds"][0]["committee_uncertainty_by_index"],
            [{"index": 1, "energy_std": 0.02, "force_std": 0.04, "score": 0.06}],
        )
