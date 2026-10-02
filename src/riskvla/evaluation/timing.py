"""Temporal metrics that require verified event and prediction timestamps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def action_lead_time_metrics(
    event_times: Sequence[float],
    prediction_times: Sequence[float],
    *,
    thresholds_seconds: Sequence[float] = (0.5, 1.0, 2.0),
) -> dict[str, Any]:
    if len(event_times) != len(prediction_times):
        raise ValueError("event_times and prediction_times must have equal length")
    if not event_times:
        raise ValueError(
            "Lead time is unsupported without verified event/prediction timestamps"
        )
    events = np.asarray(event_times, dtype=np.float64)
    predictions = np.asarray(prediction_times, dtype=np.float64)
    if not np.isfinite(events).all() or not np.isfinite(predictions).all():
        raise ValueError("All timing inputs must be finite")
    if np.any(events < 0) or np.any(predictions < 0):
        raise ValueError("Timing inputs must be non-negative")
    thresholds = [float(value) for value in thresholds_seconds]
    if any(not np.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError("Lead-time thresholds must be finite and non-negative")

    lead_times = events - predictions
    return {
        "definition": "event_time - action_prediction_time",
        "sample_count": int(lead_times.size),
        "mean_seconds": float(np.mean(lead_times)),
        "median_seconds": float(np.median(lead_times)),
        "minimum_seconds": float(np.min(lead_times)),
        "maximum_seconds": float(np.max(lead_times)),
        "positive_lead_count": int(np.sum(lead_times > 0)),
        "positive_lead_rate": float(np.mean(lead_times > 0)),
        "threshold_rates": {
            f"at_least_{threshold:g}_seconds": float(
                np.mean(lead_times >= threshold)
            )
            for threshold in thresholds
        },
        "lead_times_seconds": lead_times.tolist(),
    }


def trigger_efficiency(
    triggered: Sequence[bool],
    *,
    total_windows: int | None = None,
) -> dict[str, int | float]:
    if total_windows is None:
        total_windows = len(triggered)
    if total_windows <= 0 or len(triggered) > total_windows:
        raise ValueError("total_windows must cover every trigger decision")
    invocation_count = sum(bool(value) for value in triggered)
    return {
        "vla_invocations": invocation_count,
        "total_windows": total_windows,
        "vla_invocation_rate": invocation_count / total_windows,
    }
