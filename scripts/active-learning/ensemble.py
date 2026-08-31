"""Round-level K-fold committee training from an immutable LF checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from scripts.helper import _load_model, _train_k_fold_models


@dataclass(frozen=True)
class CommitteeTraining:
    """Saved committee artifacts and metadata for one acquisition round."""

    round_number: int
    model_prefix: str
    model_paths: dict[str, str]
    training: dict[str, Any]


def train_round_committee(
    *,
    lf_checkpoint: Path,
    acquired_atoms: list[Any],
    evaluation_atoms: list[Any],
    round_number: int,
    run_dir: Path,
    data_builder_class: Any,
    trainer_class: Any,
    tester: Any,
    loss_fn: torch.nn.Module,
    device: torch.device,
    seed: int,
    k: int,
    r_max: float,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    trainer_options: dict[str, Any],
    energy_key: str,
    forces_key: str,
    e0s: dict[str, float] | None,
    checkpoint_epochs: int | None,
    strategy: str,
    strategy_kwargs: dict[str, Any],
    generate_plots: bool,
    on_fold_complete: Callable[[dict[str, Any]], None] | None = None,
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> CommitteeTraining:
    """Train one committee round from the unchanged supplied LF checkpoint.

    The checkpoint is loaded anew for every call. `_train_k_fold_models` then
    deep-copies that pristine model for its individual folds, avoiding any
    dependency on a previous active-learning round's checkpoints.
    """
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 0:
        raise ValueError("'round_number' must be a non-negative integer")
    if not evaluation_atoms:
        raise ValueError("At least one evaluation geometry is required")
    model_prefix = f"committee_round_{round_number:03d}"
    initial_model = _load_model(lf_checkpoint, device)
    training = _train_k_fold_models(
        initial_model=initial_model,
        all_atoms=acquired_atoms,
        test_atoms=evaluation_atoms,
        model_prefix=model_prefix,
        run_dir=run_dir,
        data_builder_class=data_builder_class,
        trainer_class=trainer_class,
        tester=tester,
        loss_fn=loss_fn,
        device=device,
        seed=seed,
        k=k,
        r_max=r_max,
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        trainer_options=trainer_options,
        energy_key=energy_key,
        forces_key=forces_key,
        e0s=e0s,
        checkpoint_epochs=checkpoint_epochs,
        strategy=strategy,
        strategy_kwargs=strategy_kwargs,
        generate_plots=generate_plots,
        progress_label=f"[active-learning] round {round_number + 1}",
        on_fold_complete=on_fold_complete,
        on_checkpoint=on_checkpoint,
    )
    return CommitteeTraining(
        round_number=round_number,
        model_prefix=model_prefix,
        model_paths=dict(training["model_paths"]),
        training=training,
    )
