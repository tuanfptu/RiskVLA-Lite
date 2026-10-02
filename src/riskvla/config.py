"""Configuration loading with explicit, shallow-file inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from riskvla.constants import ACTIONS


class ConfigError(ValueError):
    """Raised when a project configuration is malformed."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and recursively resolve an optional relative ``_base_`` file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")

    base_name = loaded.pop("_base_", None)
    if base_name is None:
        return loaded
    if not isinstance(base_name, str) or not base_name:
        raise ConfigError("_base_ must be a non-empty relative path")

    base_path = (config_path.parent / base_name).resolve()
    if config_path == base_path:
        raise ConfigError("Configuration cannot inherit from itself")
    return _deep_merge(load_config(base_path), loaded)


VARIANT_FEATURES: dict[str, tuple[str, ...]] = {
    "visual_only": (),
    "visual_plus_current_risk": ("current_risk",),
    "visual_plus_temporal_risk": ("current_risk", "risk_slope", "recent_peak"),
}


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Validate one resolved A/B/C experiment configuration."""
    try:
        variant = config["vla"]["variant"]
        risk = config["risk"]
        enabled = risk["enabled"]
        features = tuple(risk["features"])
    except (KeyError, TypeError) as exc:
        raise ConfigError("Experiment config is missing vla/risk fields") from exc
    if variant not in VARIANT_FEATURES:
        raise ConfigError(f"Unknown experiment variant: {variant!r}")
    severity = config.get("evaluation", {}).get("severity")
    if severity is not None and set(severity) != set(ACTIONS):
        raise ConfigError(
            "evaluation.severity must define exactly the primary four actions"
        )
    frame_count = config.get("vision", {}).get("frame_count")
    if frame_count is not None and frame_count != 4:
        raise ConfigError("The primary frame budget is 4; 8 frames is a later experiment")
    expected = VARIANT_FEATURES[variant]
    if variant == "visual_only":
        if enabled or features:
            raise ConfigError("visual_only must disable risk features")
        return
    if not enabled:
        raise ConfigError(f"{variant} must enable risk features")
    if features != expected:
        raise ConfigError(
            f"{variant} features must be {list(expected)}, got {list(features)}"
        )
