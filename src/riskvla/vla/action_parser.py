"""Strict parsing for short action-only model output."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from riskvla.constants import ACTIONS


class InvalidActionOutput(ValueError):
    """Raised when model text does not satisfy the declared action schema."""

    def __init__(self, message: str, *, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


@dataclass(frozen=True)
class ActionPrediction:
    action: str
    confidence: float
    raw_output: str
    hazard_type: str | None = None
    ego_relation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_action_output(
    raw_output: str,
    *,
    allow_diagnostics: bool = False,
) -> ActionPrediction:
    """Parse exactly one JSON object; never coerce unknown or malformed output."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise InvalidActionOutput("Model output is empty", raw_output=str(raw_output))
    stripped = raw_output.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise InvalidActionOutput(
            f"Output is not valid JSON: {exc.msg}", raw_output=raw_output
        ) from exc
    if not isinstance(payload, dict):
        raise InvalidActionOutput("Output must be one JSON object", raw_output=raw_output)

    required = {"action", "confidence"}
    diagnostic = {"hazard_type", "ego_relation"}
    allowed = required | (diagnostic if allow_diagnostics else set())
    missing = required - set(payload)
    extras = set(payload) - allowed
    if missing:
        raise InvalidActionOutput(
            f"Missing required keys: {sorted(missing)}", raw_output=raw_output
        )
    if extras:
        raise InvalidActionOutput(
            f"Unexpected keys: {sorted(extras)}", raw_output=raw_output
        )

    action = payload["action"]
    if action not in ACTIONS:
        raise InvalidActionOutput(
            f"Unknown action {action!r}; expected one of {ACTIONS}",
            raw_output=raw_output,
        )
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise InvalidActionOutput(
            "confidence must be a number", raw_output=raw_output
        )
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise InvalidActionOutput(
            "confidence must be finite and within [0, 1]", raw_output=raw_output
        )

    values: dict[str, str | None] = {}
    for field in diagnostic:
        value = payload.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise InvalidActionOutput(
                f"{field} must be a non-empty string or null", raw_output=raw_output
            )
        values[field] = value.strip() if isinstance(value, str) else None

    return ActionPrediction(
        action=action,
        confidence=confidence,
        raw_output=raw_output,
        hazard_type=values["hazard_type"],
        ego_relation=values["ego_relation"],
    )
