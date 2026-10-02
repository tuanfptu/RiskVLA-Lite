"""Deterministic first-pass categorization for paired error review."""

from __future__ import annotations

from collections.abc import Mapping

from riskvla.constants import ACTIONS, DEFAULT_SEVERITY


def categorize_failure(
    *,
    target: str,
    prediction: str | None,
    risk_triggered: bool | None = None,
    event_time: float | None = None,
    prediction_time: float | None = None,
    frame_times: list[float] | None = None,
    severity: Mapping[str, int] | None = None,
) -> list[str]:
    if target not in ACTIONS:
        raise ValueError(f"Unknown target action: {target}")
    if prediction is not None and prediction not in ACTIONS:
        raise ValueError(f"Unknown predicted action: {prediction}")
    severity_map = dict(severity or DEFAULT_SEVERITY)
    categories: list[str] = []

    if prediction is None:
        categories.append("invalid structured output")
    elif prediction != target:
        if severity_map[prediction] < severity_map[target]:
            categories.append("under-reaction")
        elif severity_map[prediction] > severity_map[target]:
            categories.append("over-reaction")
        else:
            categories.append("VLA action-selection error")
    if target == "BRAKE_OR_STOP" and prediction != target:
        categories.append("missed critical action")
    if target != "BRAKE_OR_STOP" and prediction == "BRAKE_OR_STOP":
        categories.append("unnecessary braking")

    if event_time is not None:
        if prediction_time is None:
            raise ValueError("prediction_time is required when event_time is provided")
        if prediction == target and prediction_time >= event_time:
            categories.append("late correct action")
        if frame_times is not None and any(time >= event_time for time in frame_times):
            categories.append("post-event leakage")
    if risk_triggered is False and target == "BRAKE_OR_STOP":
        categories.append("risk detector false negative")
    if risk_triggered is True and target == "MAINTAIN":
        categories.append("risk detector false positive")
    return categories
