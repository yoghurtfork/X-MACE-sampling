"""Strict configuration loading for the unified training entry point.

This module deliberately has no MACE, ASE, or torch dependency.  A
configuration can therefore be validated before a scheduler reserves a GPU or
the training modules import their scientific dependencies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a training JSON file does not match the supported schema."""


_MODES = {"lf", "hf", "transfer", "both"}
_STRATEGIES = {"naive", "freeze"}
DEFAULT_SEED = 42
DEFAULT_DEVICE = "cuda"
DEFAULT_MAX_EPOCHS = 100
DEFAULT_R_MAX = 5.0
DEFAULT_BATCH_SIZE = 64
DEFAULT_TRANSFER_LEARNING_RATE = 5.0e-4
_LOSS_DEFAULTS = {
    "energy_weight": 1.0,
    "forces_weight": 1.0,
    "dipoles_weight": 0.0,
    "nacs_weight": 0.0,
    "socs_weight": 0.0,
}
_DEFAULTS: dict[str, Any] = {
    "ignore": False,
    "seed": DEFAULT_SEED,
    "device": DEFAULT_DEVICE,
    "cross_validation": False,
    "validation_fraction": 0.1,
    "batch_size": DEFAULT_BATCH_SIZE,
    "r_max": DEFAULT_R_MAX,
    "lf_max_epochs": DEFAULT_MAX_EPOCHS,
    "hf_max_epochs": DEFAULT_MAX_EPOCHS,
    "transfer_max_epochs": DEFAULT_MAX_EPOCHS,
    "lf_learning_rate": 1.0e-3,
    "hf_learning_rate": 1.0e-3,
    "transfer_learning_rate": DEFAULT_TRANSFER_LEARNING_RATE,
    "foundation_model": "ani500k",
    "strategy": "naive",
    "strategy_kwargs": {},
    "checkpoint_epochs": None,
    "preset": "default_ani",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "early_stopping": True,
    "restore_best": True,
    "verbose": True,
    "generate_plots": False,
    "pca": False,
    "descriptor_kwargs": {},
    "selector_kwargs": {},
}
_PATH_FIELDS = {
    "lf_xyz",
    "hf_xyz",
    "pretrained_model_path",
}
_TEST_PATH_FIELDS = {"lf_test_xyz", "hf_test_xyz"}
_OPTIONAL_COUNT_FIELDS = {"lf_n_geometries", "hf_n_geometries"}
_OPTIONAL_E0_FIELDS = {"lf_E0s", "hf_E0s"}
_TRAINER_NUMBER_FIELDS = {
    "optimiser_lr": (0.0, True),
    "optimiser_weight_decay": (0.0, False),
    "max_grad_norm": (0.0, True),
    "scheduler_lr_factor": (0.0, True),
}
_ALLOWED_FIELDS = set(_DEFAULTS) | _PATH_FIELDS | _TEST_PATH_FIELDS | _OPTIONAL_COUNT_FIELDS | _OPTIONAL_E0_FIELDS | {
    "mode",
    "k",
    "loss_kwargs",
    "descriptor",
    "selector",
    "n_samples",
    "ema_decay",
    "scheduler_patience",
    "stopping_patience",
} | set(_TRAINER_NUMBER_FIELDS)


def resolve_path(value: Any, key: str, config_path: Path) -> Path:
    """Validate and resolve a configuration path without requiring it to exist."""
    if not isinstance(value, str) or not value:
        raise ConfigError(f"'{key}' must be a non-empty path string")
    candidate = Path(value).expanduser()
    return (
        (Path(config_path).resolve().parent / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def parse_e0s(value: Any, key: str) -> dict[str, float] | None:
    """Validate an optional element-to-E0 mapping and normalize its values."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a JSON object or null")
    result: dict[str, float] = {}
    for element, energy in value.items():
        if not isinstance(element, str) or not element:
            raise ConfigError(f"'{key}' element names must be non-empty strings")
        result[element] = _number(energy, f"'{key}.{element}'")
    return result


def parse_strategy(value: Any) -> str:
    """Validate a supported MACE training strategy."""
    if not isinstance(value, str):
        raise ConfigError("'strategy' must be a JSON string")
    if value not in _STRATEGIES:
        raise ConfigError("'strategy' must be either 'naive' or 'freeze'")
    return value


def parse_strategy_kwargs(strategy: str, value: Any) -> dict[str, Any]:
    """Validate strategy-specific configuration while preserving freeze defaults."""
    if strategy not in _STRATEGIES:
        raise ConfigError("'strategy' must be either 'naive' or 'freeze'")
    if not isinstance(value, dict):
        raise ConfigError("'strategy_kwargs' must be a JSON object")
    if strategy == "naive":
        if value:
            raise ConfigError("'strategy_kwargs' is not supported for 'naive'")
        return {}
    invalid = sorted(set(value).difference({"frozen_layers"}))
    if invalid:
        raise ConfigError("'strategy_kwargs' for 'freeze' may contain only 'frozen_layers'")
    layers = value.get("frozen_layers", ["node_embedding", "interactions"])
    if not isinstance(layers, list) or not all(
        isinstance(layer, str) and layer for layer in layers
    ):
        raise ConfigError(
            "'strategy_kwargs.frozen_layers' must be an array of non-empty strings"
        )
    return {"frozen_layers": list(layers)}


def parse_checkpoint_epochs(value: Any) -> int | None:
    """Validate an optional positive checkpoint interval."""
    return None if value is None else _positive_integer(value, "'checkpoint_epochs'")


def validate_device_name(value: Any) -> str:
    """Validate a device name without importing torch or probing CUDA."""
    if value not in {"cpu", "cuda"}:
        raise ConfigError("'device' must be either 'cpu' or 'cuda'")
    return value


def parse_trainer_options(config: dict[str, Any]) -> dict[str, Any]:
    """Extract validated MACE trainer options from any compatible config object."""
    options: dict[str, Any] = {}
    for key, default in (
        ("early_stopping", True),
        ("restore_best", True),
        ("verbose", True),
    ):
        value = config.get(key, default)
        _require_bool(value, repr(key))
        options[key] = value
    for key, (minimum, strict) in _TRAINER_NUMBER_FIELDS.items():
        if key in config:
            options[key] = _number(config[key], repr(key), minimum, strict)
    if "ema_decay" in config:
        value = config["ema_decay"]
        if value is None:
            options["ema_decay"] = None
        else:
            parsed = _number(value, "'ema_decay'", 0.0, True)
            if parsed >= 1.0:
                raise ConfigError("'ema_decay' must be less than 1")
            options["ema_decay"] = parsed
    if "scheduler_patience" in config:
        options["scheduler_patience"] = _integer(
            config["scheduler_patience"], "'scheduler_patience'", minimum=0
        )
    patience_key = (
        "stopping_patience" if "stopping_patience" in config
        else "patience" if "patience" in config else None
    )
    if patience_key is not None:
        options["stopping_patience"] = _positive_integer(
            config[patience_key], repr(patience_key)
        )
    return options


def load_config(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Load one strict training configuration and return it with warnings.

    Paths in the returned dictionary are absolute :class:`~pathlib.Path`
    objects.  ``state.py`` will serialize them when recording the effective
    configuration.  The caller is responsible for emitting the returned
    warnings, which lets them be both printed and persisted.
    """
    path = Path(config_path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Configuration file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON in configuration file: {path}") from error
    if not isinstance(raw, dict):
        raise ConfigError("The top-level JSON value must be an object")

    unknown = sorted(set(raw).difference(_ALLOWED_FIELDS))
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise ConfigError(f"Unknown configuration field(s): {names}")
    mode = _required_string(raw, "mode")
    if mode not in _MODES:
        raise ConfigError("'mode' must be one of: lf, hf, transfer, both")

    config: dict[str, Any] = {**_DEFAULTS, "mode": mode}
    for key, value in raw.items():
        config[key] = value

    _validate_common(config)
    _normalize_paths(config, path)
    _normalize_optional_fields(config)
    warnings = _mode_warnings_and_requirements(config, set(raw))
    return config, warnings


def _validate_common(config: dict[str, Any]) -> None:
    _require_bool(config["ignore"], "'ignore'")
    config["seed"] = _integer(config["seed"], "'seed'")
    config["device"] = validate_device_name(config["device"])
    _require_bool(config["cross_validation"], "'cross_validation'")
    config["batch_size"] = _positive_integer(config["batch_size"], "'batch_size'")
    config["r_max"] = _positive_number(config["r_max"], "'r_max'")
    for key in ("lf_max_epochs", "hf_max_epochs", "transfer_max_epochs"):
        config[key] = _positive_integer(config[key], repr(key))
    for key in ("lf_learning_rate", "hf_learning_rate", "transfer_learning_rate"):
        config[key] = _positive_number(config[key], repr(key))
    if config["cross_validation"]:
        if "k" not in config:
            raise ConfigError("'k' is required when 'cross_validation' is true")
        config["k"] = _positive_integer(config["k"], "'k'", minimum=2)
    elif "k" in config:
        config["k"] = _positive_integer(config["k"], "'k'", minimum=2)
    else:
        config["k"] = None
    config["validation_fraction"] = _fraction(
        config["validation_fraction"], "'validation_fraction'"
    )

    foundation = config["foundation_model"]
    if foundation is not None and (not isinstance(foundation, str) or not foundation):
        raise ConfigError("'foundation_model' must be a non-empty string or null")
    config["strategy"] = parse_strategy(config["strategy"])
    config["strategy_kwargs"] = parse_strategy_kwargs(
        config["strategy"], config["strategy_kwargs"]
    )
    config["checkpoint_epochs"] = parse_checkpoint_epochs(config["checkpoint_epochs"])
    config["preset"] = _required_string(config, "preset")
    config["energy_key"] = _required_string(config, "energy_key")
    config["forces_key"] = _required_string(config, "forces_key")
    trainer_options = parse_trainer_options(config)
    config.update(trainer_options)
    for key in ("generate_plots", "pca"):
        _require_bool(config[key], repr(key))
    config["loss_kwargs"] = _loss_kwargs(config.get("loss_kwargs", {}))


def _normalize_paths(config: dict[str, Any], config_path: Path) -> None:
    for key in _PATH_FIELDS:
        if key in config:
            config[key] = resolve_path(config[key], key, config_path)
    for key in _TEST_PATH_FIELDS:
        if key in config:
            config[key] = _test_paths(config[key], key, config_path)


def _normalize_optional_fields(config: dict[str, Any]) -> None:
    for key in _OPTIONAL_COUNT_FIELDS:
        if key in config:
            config[key] = _positive_integer(config[key], repr(key))
    for key in _OPTIONAL_E0_FIELDS:
        if key in config:
            config[key] = parse_e0s(config[key], key)
    for key in ("descriptor_kwargs", "selector_kwargs"):
        if not isinstance(config[key], dict):
            raise ConfigError(f"'{key}' must be a JSON object")
    if "descriptor" in config:
        config["descriptor"] = _required_string(config, "descriptor")
    if "selector" in config:
        config["selector"] = _required_string(config, "selector")
    if "n_samples" in config:
        config["n_samples"] = _positive_integer(config["n_samples"], "'n_samples'")
    if "pretrained_model_path" in config and not isinstance(
        config["pretrained_model_path"], Path
    ):
        raise AssertionError("pretrained_model_path was not normalized")


def _mode_warnings_and_requirements(
    config: dict[str, Any], provided: set[str]
) -> list[str]:
    mode = config["mode"]
    required_by_mode = {
        "lf": {"lf_xyz", "lf_test_xyz"},
        "hf": {"hf_xyz", "hf_test_xyz"},
        "both": {"lf_xyz", "lf_test_xyz", "hf_xyz", "hf_test_xyz"},
        "transfer": {
            "pretrained_model_path", "lf_xyz", "hf_xyz", "hf_test_xyz",
            "descriptor", "selector", "n_samples",
        },
    }
    missing = sorted(required_by_mode[mode].difference(config))
    if missing:
        raise ConfigError(
            "Missing required configuration field(s) for "
            f"'{mode}' mode: {', '.join(repr(key) for key in missing)}"
        )

    unused: set[str]
    if mode == "lf":
        unused = _hf_stage_fields() | _transfer_fields()
    elif mode == "hf":
        unused = _lf_stage_fields() | _transfer_fields()
    elif mode == "both":
        unused = _transfer_fields()
    else:
        unused = {
            "lf_test_xyz", "foundation_model", "lf_max_epochs",
            "hf_max_epochs", "lf_learning_rate", "hf_learning_rate",
        }
    warnings = [
        f"Ignoring '{key}' because it is not used in '{mode}' mode"
        for key in sorted(unused.intersection(config))
        if key in provided
    ]
    if config["cross_validation"] and "validation_fraction" in provided:
        warnings.append("Ignoring 'validation_fraction' because cross-validation is enabled")
    if not config["cross_validation"] and "k" in provided:
        warnings.append("Ignoring 'k' because cross-validation is disabled")
    return warnings


def _lf_stage_fields() -> set[str]:
    return {"lf_xyz", "lf_test_xyz", "lf_n_geometries", "lf_E0s", "lf_max_epochs", "lf_learning_rate"}


def _hf_stage_fields() -> set[str]:
    return {"hf_xyz", "hf_test_xyz", "hf_n_geometries", "hf_E0s", "hf_max_epochs", "hf_learning_rate"}


def _transfer_fields() -> set[str]:
    return {"pretrained_model_path", "descriptor", "descriptor_kwargs", "selector", "selector_kwargs", "n_samples", "pca", "transfer_max_epochs", "transfer_learning_rate"}


def _path(value: Any, key: str, config_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"'{key}' must be a non-empty path string")
    candidate = Path(value).expanduser()
    return (config_path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _test_paths(value: Any, key: str, config_path: Path) -> list[Path]:
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not values:
        raise ConfigError(f"'{key}' must be a path string or a non-empty array of path strings")
    return [resolve_path(item, key, config_path) for item in values]


def _loss_kwargs(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ConfigError("'loss_kwargs' must be a JSON object")
    invalid = sorted(set(value).difference(_LOSS_DEFAULTS))
    if invalid:
        raise ConfigError("Unsupported 'loss_kwargs' entries: " + ", ".join(repr(key) for key in invalid))
    return {key: _number(value.get(key, default), f"'loss_kwargs.{key}'", 0.0, False) for key, default in _LOSS_DEFAULTS.items()}


def _strategy_kwargs(strategy: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError("'strategy_kwargs' must be a JSON object")
    if strategy == "naive":
        if value:
            raise ConfigError("'strategy_kwargs' is not supported for 'naive'")
        return {}
    invalid = sorted(set(value).difference({"frozen_layers"}))
    if invalid:
        raise ConfigError("'strategy_kwargs' for 'freeze' may contain only 'frozen_layers'")
    layers = value.get("frozen_layers", ["node_embedding", "interactions"])
    if not isinstance(layers, list) or not all(isinstance(layer, str) and layer for layer in layers):
        raise ConfigError("'strategy_kwargs.frozen_layers' must be an array of non-empty strings")
    return {"frozen_layers": list(layers)}


def _e0s(value: Any, key: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a JSON object")
    result: dict[str, float] = {}
    for element, energy in value.items():
        if not isinstance(element, str) or not element:
            raise ConfigError(f"'{key}' element names must be non-empty strings")
        result[element] = _number(energy, f"'{key}.{element}'")
    return result


def _required_string(mapping: dict[str, Any], key: str) -> str:
    if key not in mapping:
        raise ConfigError(f"Missing required configuration field: {key!r}")
    value = mapping[key]
    if not isinstance(value, str) or not value:
        raise ConfigError(f"'{key}' must be a non-empty string")
    return value


def _require_bool(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a JSON boolean")


def _integer(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
        comparison = "positive" if minimum == 1 else f"at least {minimum}"
        raise ConfigError(f"{name} must be a {comparison} JSON integer" if minimum is not None else f"{name} must be a JSON integer")
    return value


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    return _integer(value, name, minimum)


def _optional_positive_integer(value: Any, name: str) -> int | None:
    return None if value is None else _positive_integer(value, name)


def _fraction(value: Any, name: str) -> float:
    result = _number(value, name)
    if not 0.0 < result < 1.0:
        raise ConfigError(f"{name} must be between 0 and 1")
    return result


def _positive_number(value: Any, name: str) -> float:
    return _number(value, name, 0.0, True)


def _number(value: Any, name: str, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite")
    if minimum is not None and (result < minimum or (strict and result <= minimum)):
        qualifier = "greater than" if strict else "at least"
        raise ConfigError(f"{name} must be {qualifier} {minimum}")
    return result
