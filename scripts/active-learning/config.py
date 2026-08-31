"""Configuration parsing for JSON-driven active-transfer-learning runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.helper import (
    BATCH_SIZE,
    DEVICE,
    MAX_EPOCHS,
    R_MAX,
    SEED,
    TRANSFER_LR,
    _e0s_from_config,
    _path,
    _strategy_from_config,
    _strategy_kwargs_from_config,
    _trainer_options_from_config,
    _validate_device,
)


_LOSS_DEFAULTS = {
    "energy_weight": 1.0,
    "forces_weight": 5.0,
    "dipoles_weight": 0.0,
    "nacs_weight": 0.0,
    "socs_weight": 0.0,
}


@dataclass(frozen=True)
class ActiveLearningConfig:
    """Validated settings for one active-transfer-learning experiment."""

    config_path: Path
    lf_checkpoint: Path
    hf_xyz: Path
    hf_test_xyz: Path | None
    grid_shape: tuple[int, int]
    initial_acquired_count: int
    n_rounds: int
    k: int
    max_seeds_per_round: int
    uncertainty_threshold: float
    energy_uncertainty_weight: float
    force_uncertainty_weight: float
    seed: int
    device: str
    max_epochs: int
    final_max_epochs: int
    r_max: float
    batch_size: int
    learning_rate: float
    final_learning_rate: float
    energy_key: str
    forces_key: str
    e0s: dict[str, float] | None
    strategy: str
    strategy_kwargs: dict[str, Any]
    trainer_options: dict[str, Any]
    loss_kwargs: dict[str, float]
    checkpoint_epochs: int | None


def load_config(config_path: Path) -> ActiveLearningConfig:
    """Load and validate an active-learning JSON configuration file."""
    config_path = config_path.resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in configuration file: {config_path}") from error
    if not isinstance(config, dict):
        raise ValueError("The top-level JSON value must be an object")

    lf_checkpoint = _existing_file(config, "lf_checkpoint", config_path)
    hf_xyz = _existing_file(config, "hf_xyz", config_path)
    hf_test_xyz = _optional_existing_file(config, "hf_test_xyz", config_path)
    grid_shape = _grid_shape(config)
    initial_acquired_count = _positive_int(config, "initial_acquired_count")
    k = _positive_int(config, "k", minimum=2)
    if initial_acquired_count < k:
        raise ValueError("'initial_acquired_count' must be at least 'k'")

    n_rounds = _non_negative_int(config, "n_rounds")
    max_seeds_per_round = _positive_int(config, "max_seeds_per_round")
    uncertainty_threshold = _non_negative_number(config, "uncertainty_threshold")
    energy_uncertainty_weight = _non_negative_number(
        config, "energy_uncertainty_weight", default=1.0
    )
    force_uncertainty_weight = _non_negative_number(
        config, "force_uncertainty_weight", default=1.0
    )
    if energy_uncertainty_weight == 0.0 and force_uncertainty_weight == 0.0:
        raise ValueError("At least one uncertainty weight must be positive")

    seed = _int_with_default(config, "seed", SEED)
    max_epochs = _positive_int(config, "max_epochs", default=MAX_EPOCHS)
    final_max_epochs = _positive_int(
        config, "final_max_epochs", default=max_epochs
    )
    batch_size = _positive_int(config, "batch_size", default=BATCH_SIZE)
    r_max = _positive_number(config, "r_max", default=R_MAX)
    learning_rate = _positive_number(
        config, "transfer_lr", default=TRANSFER_LR
    )
    final_learning_rate = _positive_number(
        config, "final_transfer_lr", default=learning_rate
    )
    energy_key = _non_empty_string(config, "energy_key", default="REF_energy")
    forces_key = _non_empty_string(config, "forces_key", default="REF_forces")
    checkpoint_epochs = _optional_positive_int(config, "checkpoint_epochs")
    strategy = _strategy_from_config(config)

    device_name = _non_empty_string(config, "device", default=DEVICE)
    device = str(_validate_device(device_name))
    loss_kwargs = _loss_kwargs(config)

    return ActiveLearningConfig(
        config_path=config_path,
        lf_checkpoint=lf_checkpoint,
        hf_xyz=hf_xyz,
        hf_test_xyz=hf_test_xyz,
        grid_shape=grid_shape,
        initial_acquired_count=initial_acquired_count,
        n_rounds=n_rounds,
        k=k,
        max_seeds_per_round=max_seeds_per_round,
        uncertainty_threshold=uncertainty_threshold,
        energy_uncertainty_weight=energy_uncertainty_weight,
        force_uncertainty_weight=force_uncertainty_weight,
        seed=seed,
        device=device,
        max_epochs=max_epochs,
        final_max_epochs=final_max_epochs,
        r_max=r_max,
        batch_size=batch_size,
        learning_rate=learning_rate,
        final_learning_rate=final_learning_rate,
        energy_key=energy_key,
        forces_key=forces_key,
        e0s=_e0s_from_config(config, "hf_E0s"),
        strategy=strategy,
        strategy_kwargs=_strategy_kwargs_from_config(config, strategy),
        trainer_options=_trainer_options_from_config(config),
        loss_kwargs=loss_kwargs,
        checkpoint_epochs=checkpoint_epochs,
    )


def _existing_file(config: dict[str, Any], key: str, config_path: Path) -> Path:
    path = _path(_non_empty_string(config, key), config_path)
    if not path.is_file():
        raise FileNotFoundError(f"'{key}' file does not exist: {path}")
    return path


def _optional_existing_file(
    config: dict[str, Any], key: str, config_path: Path
) -> Path | None:
    if key not in config or config[key] is None:
        return None
    return _existing_file(config, key, config_path)


def _grid_shape(config: dict[str, Any]) -> tuple[int, int]:
    value = config.get("grid_shape")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("'grid_shape' must be a JSON array of two positive integers")
    dimensions = tuple(_require_int(item, "'grid_shape' entries", minimum=1) for item in value)
    return dimensions  # type: ignore[return-value]


def _loss_kwargs(config: dict[str, Any]) -> dict[str, float]:
    value = config.get("loss_kwargs", {})
    if not isinstance(value, dict):
        raise ValueError("'loss_kwargs' must be a JSON object")
    invalid_keys = set(value).difference(_LOSS_DEFAULTS)
    if invalid_keys:
        names = ", ".join(sorted(repr(key) for key in invalid_keys))
        raise ValueError(f"Unsupported 'loss_kwargs' entries: {names}")
    result = dict(_LOSS_DEFAULTS)
    for key, weight in value.items():
        result[key] = _require_number(weight, f"'loss_kwargs.{key}'", minimum=0.0)
    return result


def _positive_int(
    config: dict[str, Any], key: str, *, default: int | None = None, minimum: int = 1
) -> int:
    if key not in config:
        if default is None:
            raise ValueError(f"Missing required configuration key: {key!r}")
        return default
    return _require_int(config[key], repr(key), minimum=minimum)


def _non_negative_int(config: dict[str, Any], key: str) -> int:
    if key not in config:
        raise ValueError(f"Missing required configuration key: {key!r}")
    return _require_int(config[key], repr(key), minimum=0)


def _optional_positive_int(config: dict[str, Any], key: str) -> int | None:
    if key not in config or config[key] is None:
        return None
    return _require_int(config[key], repr(key), minimum=1)


def _int_with_default(config: dict[str, Any], key: str, default: int) -> int:
    return default if key not in config else _require_int(config[key], repr(key))


def _positive_number(
    config: dict[str, Any], key: str, *, default: float
) -> float:
    value = default if key not in config else config[key]
    return _require_number(value, repr(key), minimum=0.0, strict=True)


def _non_negative_number(
    config: dict[str, Any], key: str, *, default: float | None = None
) -> float:
    if key not in config:
        if default is None:
            raise ValueError(f"Missing required configuration key: {key!r}")
        return default
    return _require_number(config[key], repr(key), minimum=0.0)


def _non_empty_string(
    config: dict[str, Any], key: str, *, default: str | None = None
) -> str:
    if key not in config:
        if default is None:
            raise ValueError(f"Missing required configuration key: {key!r}")
        return default
    value = config[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key!r} must be a non-empty JSON string")
    return value


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a JSON integer")
    if minimum is not None and value < minimum:
        qualifier = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _require_number(
    value: Any, name: str, *, minimum: float, strict: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    number = float(value)
    if (strict and number <= minimum) or (not strict and number < minimum):
        qualifier = "positive" if strict else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return number
