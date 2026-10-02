"""Causal temporal feature extraction from normalized risk sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from riskvla.risk.base import RiskSequence

TriggerPolicy = Literal["always", "risk_threshold", "risk_or_slope"]


@dataclass(frozen=True)
class TriggerConfig:
    policy: TriggerPolicy = "risk_or_slope"
    risk_threshold: float = 0.8
    slope_threshold: float = 0.15

    def __post_init__(self) -> None:
        if self.policy not in {"always", "risk_threshold", "risk_or_slope"}:
            raise ValueError(f"Unknown trigger policy: {self.policy}")
        if not 0.0 <= self.risk_threshold <= 1.0:
            raise ValueError("risk_threshold must be in [0, 1]")
        if not np.isfinite(self.slope_threshold):
            raise ValueError("slope_threshold must be finite")


@dataclass(frozen=True)
class RiskFeatures:
    prediction_timestamp: float
    current_risk: float
    risk_slope: float
    recent_peak: float
    risk_mean: float
    risk_volatility: float
    trigger_timestamp: float | None
    triggered: bool
    valid_observations: int

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


def _least_squares_slope(
    timestamps: NDArray[np.float64], scores: NDArray[np.float32]
) -> float:
    if timestamps.size < 2:
        return 0.0
    centered_t = timestamps - timestamps.mean()
    denominator = float(np.dot(centered_t, centered_t))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    centered_scores = scores.astype(np.float64) - float(scores.mean())
    return float(np.dot(centered_t, centered_scores) / denominator)


def _window(
    timestamps: NDArray[np.float64],
    scores: NDArray[np.float32],
    *,
    end_index: int,
    seconds: float,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    end_time = timestamps[end_index]
    mask = (
        (timestamps <= end_time)
        & (timestamps >= end_time - seconds)
        & np.isfinite(scores)
    )
    return timestamps[mask], scores[mask]


def extract_risk_features(
    sequence: RiskSequence,
    *,
    prediction_timestamp: float | None = None,
    slope_window_seconds: float = 2.0,
    peak_window_seconds: float = 2.0,
    trigger: TriggerConfig | None = None,
) -> RiskFeatures:
    """Compute causal features using observations no later than prediction time."""
    if slope_window_seconds <= 0 or peak_window_seconds <= 0:
        raise ValueError("Feature windows must be positive")
    cutoff = (
        sequence.duration_seconds
        if prediction_timestamp is None
        else float(prediction_timestamp)
    )
    causal = sequence.through(cutoff)
    valid_indices = np.flatnonzero(causal.valid_mask)
    if valid_indices.size == 0:
        raise ValueError("No finite risk observation exists at or before prediction time")
    current_index = int(valid_indices[-1])
    current_time = float(causal.timestamps[current_index])
    current_risk = float(causal.risk_scores[current_index])

    slope_t, slope_scores = _window(
        causal.timestamps,
        causal.risk_scores,
        end_index=current_index,
        seconds=slope_window_seconds,
    )
    peak_t, peak_scores = _window(
        causal.timestamps,
        causal.risk_scores,
        end_index=current_index,
        seconds=peak_window_seconds,
    )
    if peak_t.size == 0:
        raise ValueError("Recent peak window has no finite observations")

    trigger_config = trigger or TriggerConfig()
    trigger_timestamp: float | None = None
    if trigger_config.policy == "always":
        trigger_timestamp = float(causal.timestamps[valid_indices[0]])
    else:
        for index in valid_indices:
            historical_t, historical_scores = _window(
                causal.timestamps,
                causal.risk_scores,
                end_index=int(index),
                seconds=slope_window_seconds,
            )
            historical_slope = _least_squares_slope(historical_t, historical_scores)
            risk_crossed = (
                float(causal.risk_scores[index]) > trigger_config.risk_threshold
            )
            slope_crossed = historical_slope > trigger_config.slope_threshold
            if risk_crossed or (
                trigger_config.policy == "risk_or_slope" and slope_crossed
            ):
                trigger_timestamp = float(causal.timestamps[index])
                break

    return RiskFeatures(
        prediction_timestamp=current_time,
        current_risk=current_risk,
        risk_slope=_least_squares_slope(slope_t, slope_scores),
        recent_peak=float(np.max(peak_scores)),
        risk_mean=float(np.mean(peak_scores)),
        risk_volatility=float(np.std(peak_scores)),
        trigger_timestamp=trigger_timestamp,
        triggered=trigger_timestamp is not None and trigger_timestamp <= current_time,
        valid_observations=int(valid_indices.size),
    )
