"""Run resumable active transfer learning from a JSON configuration file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ActiveLearningConfig, load_config
from data import HFGrid
from ensemble import train_round_committee
from predict import predict_committee
from selection import select_acquisitions
from state import (
    StateStore,
    begin_final_production_model,
    checkpoint_current_round,
    complete_current_round,
    complete_final_production_model,
    interrupt_current_round,
    make_resume_identity,
    new_state,
    restart_interrupted_round,
    start_round,
)
from uncertainty import committee_uncertainty
from scripts.helper import (
    _apply_training_strategy,
    _e0s_from_metadata,
    _evaluate,
    _import_project_modules,
    _load_model,
    _next_run_dir,
    _save_model,
    _trainer_options_for_learning_rate,
    _write_json,
    seed_everything,
)


def run_config(
    config_path: Path,
    output_dir: Path,
    *,
    run_dir: Path | None = None,
    resume: bool = False,
) -> Path:
    """Run or resume one JSON-configured active-learning experiment."""
    config = load_config(config_path)
    raw_config = _read_raw_config(config.config_path)
    grid = HFGrid.from_xyz(
        config.hf_xyz,
        grid_shape=config.grid_shape,
        energy_key=config.energy_key,
        forces_key=config.forces_key,
    )
    if config.initial_acquired_count > grid.size:
        raise ValueError("'initial_acquired_count' exceeds the HF-grid size")
    identity = make_resume_identity(
        config_path=config.config_path,
        lf_checkpoint=config.lf_checkpoint,
        hf_xyz=config.hf_xyz,
        grid_shape=config.grid_shape,
    )

    if resume:
        if run_dir is None:
            raise ValueError("A run directory is required when resuming")
        run_dir = run_dir.resolve()
        store = StateStore(run_dir / "result.json")
        state = store.resume(identity)
    else:
        if run_dir is None:
            run_dir = _next_run_dir(output_dir)
        else:
            run_dir = run_dir.resolve()
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        store = StateStore(run_dir / "result.json")
        if store.result_path.exists():
            raise FileExistsError(
                f"Result file already exists; use --resume: {store.result_path}"
            )
        initial_indices = np.sort(
            np.random.default_rng(config.seed).choice(
                grid.size, size=config.initial_acquired_count, replace=False
            )
        ).tolist()
        state = new_state(
            input_file=config.config_path,
            run_directory=run_dir,
            config=raw_config,
            resume_identity=identity,
            grid_size=grid.size,
            grid_shape=config.grid_shape,
            energy_key=config.energy_key,
            forces_key=config.forces_key,
            initial_acquired_indices=initial_indices,
        )
        store.save(state)

    (
        modules,
        AtomDataLoaderBuilder,
        Tester,
        _extract_latent_space,
        _NaiveStrategy,
        Trainer,
        _initialise_autoencoder,
        _descriptors,
        _selectors,
    ) = _import_project_modules()
    device = torch.device(config.device)
    loss_fn = modules.InvariantsWeightedEnergyForcesNacsDipoleLoss(
        **config.loss_kwargs
    ).to(device)
    hf_test_atoms = _read_test_atoms(config.hf_test_xyz)

    try:
        _run_rounds(
            state=state,
            store=store,
            config=config,
            grid=grid,
            test_atoms=hf_test_atoms,
            data_builder_class=AtomDataLoaderBuilder,
            trainer_class=Trainer,
            tester_class=Tester,
            loss_fn=loss_fn,
            device=device,
        )
        _train_final_production_model(
            state=state,
            store=store,
            config=config,
            grid=grid,
            test_atoms=hf_test_atoms,
            data_builder_class=AtomDataLoaderBuilder,
            trainer_class=Trainer,
            tester_class=Tester,
            loss_fn=loss_fn,
            device=device,
            run_dir=run_dir,
        )
    except KeyboardInterrupt:
        _mark_interrupted(state)
        store.save(state)
        raise
    except Exception as error:
        state["status"] = "failed"
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        store.save(state)
        raise
    return store.result_path


def _run_rounds(
    *,
    state: dict[str, Any],
    store: StateStore,
    config: ActiveLearningConfig,
    grid: HFGrid,
    test_atoms: list[Any] | None,
    data_builder_class: Any,
    trainer_class: Any,
    tester_class: Any,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> None:
    retry_round: int | None = None
    if (
        state["status"] == "interrupted"
        and state["rounds"]
        and state["rounds"][-1]["status"] == "interrupted"
    ):
        retry_round = restart_interrupted_round(state)["round_number"]
        store.save(state)
    elif state["status"] == "interrupted":
        state["status"] = "running"
        store.save(state)
    completed = {
        record["round_number"]
        for record in state["rounds"]
        if record["status"] == "completed"
    }
    for round_number in range(config.n_rounds):
        if round_number in completed:
            continue
        if retry_round != round_number:
            start_round(
                state,
                round_number=round_number,
                acquired_before=state["acquired_indices"],
            )
            store.save(state)
        _run_one_round(
            state=state,
            store=store,
            config=config,
            grid=grid,
            test_atoms=test_atoms,
            data_builder_class=data_builder_class,
            trainer_class=trainer_class,
            tester_class=tester_class,
            loss_fn=loss_fn,
            device=device,
            round_number=round_number,
        )
        if state.get("termination_reason"):
            return


def _run_one_round(
    *,
    state: dict[str, Any],
    store: StateStore,
    config: ActiveLearningConfig,
    grid: HFGrid,
    test_atoms: list[Any] | None,
    data_builder_class: Any,
    trainer_class: Any,
    tester_class: Any,
    loss_fn: torch.nn.Module,
    device: torch.device,
    round_number: int,
) -> None:
    acquired_atoms = grid.reveal(state["acquired_indices"])
    evaluation_atoms = test_atoms if test_atoms is not None else acquired_atoms
    evaluation_source = "hf_test_xyz" if test_atoms is not None else "acquired_hf"

    def checkpoint(snapshot: dict[str, Any]) -> None:
        checkpoint_current_round(state, committee=snapshot)
        store.save(state)

    committee = train_round_committee(
        lf_checkpoint=config.lf_checkpoint,
        acquired_atoms=acquired_atoms,
        evaluation_atoms=evaluation_atoms,
        round_number=round_number,
        run_dir=Path(state["run_directory"]),
        data_builder_class=data_builder_class,
        trainer_class=trainer_class,
        tester=tester_class(device=device),
        loss_fn=loss_fn,
        device=device,
        seed=config.seed,
        k=config.k,
        r_max=config.r_max,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        trainer_options=config.trainer_options,
        energy_key=config.energy_key,
        forces_key=config.forces_key,
        e0s=config.e0s,
        checkpoint_epochs=config.checkpoint_epochs,
        strategy=config.strategy,
        strategy_kwargs=config.strategy_kwargs,
        generate_plots=config.trainer_options["verbose"],
        on_fold_complete=checkpoint,
        on_checkpoint=checkpoint,
    )
    unacquired = grid.unacquired_indices(state["acquired_indices"])
    if not len(unacquired):
        _complete_round(
            state, store, committee.training, evaluation_source, unacquired,
            np.asarray([], dtype=float), grid, config,
        )
        state["termination_reason"] = "HF grid is fully acquired"
        store.save(state)
        return

    prediction_builder = data_builder_class(
        cutoff=config.r_max,
        energy_key=config.energy_key,
        forces_key=config.forces_key,
        E0s=committee.training["E0s"],
    )
    prediction_loader = prediction_builder.load(
        grid.prediction_atoms(unacquired), batch_size=config.batch_size, shuffle=False
    )
    models = [
        _load_model(Path(path), device)
        for path in committee.model_paths.values()
    ]
    predictions = predict_committee(models, prediction_loader, device=device)
    uncertainty = committee_uncertainty(
        predictions.energies,
        predictions.forces,
        energy_weight=config.energy_uncertainty_weight,
        force_weight=config.force_uncertainty_weight,
    )
    all_scores = np.zeros(grid.size, dtype=float)
    all_scores[unacquired] = uncertainty.score
    selection = select_acquisitions(
        all_scores,
        grid.coordinates,
        already_acquired=state["acquired_indices"],
        grid_shape=config.grid_shape,
        uncertainty_threshold=config.uncertainty_threshold,
        max_seeds=config.max_seeds_per_round,
    )
    _complete_round(
        state, store, committee.training, evaluation_source, unacquired,
        uncertainty.score, grid, config, selection=selection,
    )
    if not len(selection.seed_indices):
        state["termination_reason"] = "No unacquired point exceeded the uncertainty threshold"
        store.save(state)


def _complete_round(
    state: dict[str, Any],
    store: StateStore,
    committee: dict[str, Any],
    evaluation_source: str,
    unacquired: np.ndarray,
    scores: np.ndarray,
    grid: HFGrid,
    config: ActiveLearningConfig,
    *,
    selection: Any | None = None,
) -> None:
    if selection is None:
        selection_summary = {
            "eligible_indices": [],
            "seed_indices": [],
            "neighbour_indices": [],
            "acquired_indices": [],
        }
        acquired_after = state["acquired_indices"]
    else:
        selection_summary = {
            "eligible_indices": selection.eligible_indices.tolist(),
            "seed_indices": selection.seed_indices.tolist(),
            "neighbour_indices": selection.neighbour_indices.tolist(),
            "acquired_indices": selection.acquired_indices.tolist(),
        }
        acquired_after = sorted(
            set(state["acquired_indices"]).union(selection.acquired_indices.tolist())
        )
    uncertainty_summary = {
        "candidate_count": int(len(unacquired)),
        "eligible_count": int(len(selection_summary["eligible_indices"])),
        "score_min": float(np.min(scores)) if len(scores) else None,
        "score_max": float(np.max(scores)) if len(scores) else None,
        "score_mean": float(np.mean(scores)) if len(scores) else None,
    }
    complete_current_round(
        state,
        committee={**committee, "evaluation_source": evaluation_source},
        uncertainty=uncertainty_summary,
        selection=selection_summary,
        acquired_after=acquired_after,
    )
    store.save(state)


def _train_final_production_model(
    *,
    state: dict[str, Any],
    store: StateStore,
    config: ActiveLearningConfig,
    grid: HFGrid,
    test_atoms: list[Any] | None,
    data_builder_class: Any,
    trainer_class: Any,
    tester_class: Any,
    loss_fn: torch.nn.Module,
    device: torch.device,
    run_dir: Path,
) -> None:
    if state["final_production_model"].get("status") == "completed":
        return
    begin_final_production_model(state)
    store.save(state)
    acquired_atoms = grid.reveal(state["acquired_indices"])
    seed_everything(config.seed)
    builder = data_builder_class(
        cutoff=config.r_max,
        energy_key=config.energy_key,
        forces_key=config.forces_key,
        E0s=config.e0s,
    )
    train_loader = builder.load(
        acquired_atoms, batch_size=config.batch_size, shuffle=True
    )
    model = _apply_training_strategy(
        _load_model(config.lf_checkpoint, device),
        config.strategy,
        config.strategy_kwargs,
    ).to(device)
    trainer = trainer_class(
        max_epochs=config.final_max_epochs,
        device=device,
        **_trainer_options_for_learning_rate(
            {**config.trainer_options, "early_stopping": False, "restore_best": False},
            config.final_learning_rate,
        ),
    )
    from mace.training.trainer import build_optimiser

    optimiser = build_optimiser(
        model,
        lr=config.final_learning_rate,
        weight_decay=trainer.optimiser_weight_decay,
    )
    history: dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "train_energy_mae": [],
        "train_force_mae": [],
        "learning_rate": [],
    }
    started_at = time.time()
    for epoch in range(1, config.final_max_epochs + 1):
        learning_rate = optimiser.param_groups[0]["lr"]
        metrics = trainer._run_epoch(
            model, train_loader, optimiser, loss_fn, training=True
        )
        history["epoch"].append(epoch)
        history["train_loss"].append(float(metrics["loss"]))
        history["train_energy_mae"].append(float(metrics["energy_mae"]))
        history["train_force_mae"].append(float(metrics["force_mae"]))
        history["learning_rate"].append(float(learning_rate))
    history["stopped_epoch"] = config.final_max_epochs
    model_path = _save_model(model, run_dir / "final_production_model.pt")
    result: dict[str, Any] = {
        "model_path": str(model_path),
        "history": history,
        "training_seconds": time.time() - started_at,
        "E0s": _e0s_from_metadata(builder.get_metadata()),
        "validation": "disabled",
        "early_stopping": "disabled",
        "scheduler": "disabled",
        "restore_best": "disabled",
    }
    if test_atoms is not None:
        test_loader = builder.load(
            test_atoms, batch_size=config.batch_size, shuffle=False
        )
        result["hf_test_metrics"] = _evaluate(
            model, test_loader, tester_class(device=device)
        )
    complete_final_production_model(state, result=result)
    store.save(state)


def _mark_interrupted(state: dict[str, Any]) -> None:
    if state.get("rounds") and state["rounds"][-1].get("status") == "training":
        interrupt_current_round(state)
        return
    state["final_production_model"] = {"status": "interrupted"}
    state["status"] = "interrupted"


def _read_raw_config(config_path: Path) -> dict[str, Any]:
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("The top-level JSON value must be an object")
    return value


def _read_test_atoms(path: Path | None) -> list[Any] | None:
    if path is None:
        return None
    from scripts.helper import _read_atoms

    return _read_atoms(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="active-learning JSON file")
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "output",
        help="directory for new run folders",
    )
    parser.add_argument(
        "--resume", type=Path, metavar="RUN_DIR",
        help="resume an interrupted run directory",
    )
    args = parser.parse_args()
    try:
        result_path = run_config(
            args.config,
            args.output_dir,
            run_dir=args.resume,
            resume=args.resume is not None,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Active-learning run failed: {error}", file=sys.stderr)
        return 1
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
