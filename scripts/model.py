"""Model, loss, strategy, and trainer construction for unified training."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_device(name: str) -> Any:
    """Return a usable torch device, failing clearly for unavailable CUDA."""
    import torch

    if name not in {"cpu", "cuda"}:
        raise ValueError("'device' must be either 'cpu' or 'cuda'")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def build_loss(config: dict[str, Any], device: Any) -> Any:
    """Construct the configured X-MACE energy/forces loss on ``device``."""
    from mace import modules

    return modules.InvariantsWeightedEnergyForcesNacsDipoleLoss(
        **config["loss_kwargs"]
    ).to(device)


def build_scratch_model(metadata: Any, config: dict[str, Any], device: Any) -> Any:
    """Initialize a new scratch model from the configured foundation model."""
    from mace.training import initialise_autoencoder

    return initialise_autoencoder(
        metadata,
        preset=config["preset"],
        load_base=config["foundation_model"],
    ).to(device)


def load_model(model_path: Path, device: Any) -> Any:
    """Load one persisted model onto the requested device."""
    import torch

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    return torch.load(path, map_location=device, weights_only=False).to(device)


def apply_training_strategy(model: Any, config: dict[str, Any]) -> Any:
    """Apply the configured naive or freeze strategy to a model."""
    from mace.training.strategies import FreezeStrategy, NaiveStrategy

    strategy = config["strategy"]
    strategy_kwargs = config["strategy_kwargs"]
    if strategy == "naive":
        return NaiveStrategy().apply(model)
    if strategy == "freeze":
        return FreezeStrategy(
            frozen_layers=tuple(strategy_kwargs["frozen_layers"])
        ).apply(model)
    # ``config.load_config`` validates this; retain a local guard for callers
    # using these helpers programmatically.
    raise ValueError("'strategy' must be either 'naive' or 'freeze'")


def build_trainer(config: dict[str, Any], stage: str, device: Any) -> Any:
    """Construct the configured X-MACE trainer for an LF, HF, or transfer stage."""
    from mace.training import Trainer

    if stage not in {"lf", "hf", "transfer"}:
        raise ValueError("'stage' must be one of: lf, hf, transfer")
    options = trainer_options(config)
    options["optimiser_lr"] = config[f"{stage}_learning_rate"]
    return Trainer(max_epochs=config[f"{stage}_max_epochs"], device=device, **options)


def trainer_options(config: dict[str, Any]) -> dict[str, Any]:
    """Extract only X-MACE ``Trainer`` options from normalized configuration."""
    options = {
        "early_stopping": config["early_stopping"],
        "restore_best": config["restore_best"],
        "verbose": config["verbose"],
    }
    for key in (
        "optimiser_lr",
        "optimiser_weight_decay",
        "max_grad_norm",
        "scheduler_lr_factor",
        "scheduler_patience",
        "ema_decay",
        "stopping_patience",
    ):
        if key in config:
            options[key] = config[key]
    return options
