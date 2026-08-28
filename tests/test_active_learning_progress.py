"""Regression tests for active-learning terminal progress output."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_LEARNING_DIR = PROJECT_ROOT / "scripts" / "active-learning"
if str(ACTIVE_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVE_LEARNING_DIR))

import run as active_learning_run


class ActiveLearningProgressTests(unittest.TestCase):
    def test_round_start_message_includes_progress_and_pool_sizes(self) -> None:
        state = {
            "status": "running",
            "rounds": [],
            "acquired_indices": [0, 1],
            "hf_grid": {"size": 5},
        }
        store = Mock()
        config = SimpleNamespace(n_rounds=1)
        grid = SimpleNamespace(size=5)
        with (
            patch.object(active_learning_run, "_run_one_round") as run_one_round,
            patch("builtins.print") as output,
        ):
            active_learning_run._run_rounds(
                state=state,
                store=store,
                config=config,
                grid=grid,
                test_atoms=None,
                data_builder_class=object(),
                trainer_class=object(),
                tester_class=object(),
                loss_fn=object(),
                device=object(),
            )

        run_one_round.assert_called_once()
        output.assert_called_once_with(
            "[active-learning] starting round 1/1 (acquired=2, unacquired=3)",
            flush=True,
        )

    def test_final_epoch_message_respects_verbose(self) -> None:
        metrics = {"loss": 0.25, "energy_mae": 0.1, "force_mae": 0.2}
        with patch("builtins.print") as output:
            active_learning_run._print_final_epoch(
                epoch=1,
                max_epochs=2,
                metrics=metrics,
                learning_rate=0.001,
                verbose=False,
            )
        output.assert_not_called()

        with patch("builtins.print") as output:
            active_learning_run._print_final_epoch(
                epoch=1,
                max_epochs=2,
                metrics=metrics,
                learning_rate=0.001,
                verbose=True,
            )
        output.assert_called_once_with(
            "[active-learning] final production | epoch 1/2 | "
            "train_loss=0.250000 | energy_mae=0.100000 | "
            "force_mae=0.200000 | lr=1.00e-03",
            flush=True,
        )
