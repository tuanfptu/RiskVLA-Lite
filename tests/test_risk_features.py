from __future__ import annotations

import numpy as np
import pytest

from riskvla.risk.base import RiskSequence
from riskvla.risk.features import TriggerConfig, extract_risk_features


def test_linear_risk_slope_and_peak() -> None:
    timestamps = np.arange(0.0, 5.0, 1.0)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    features = extract_risk_features(
        RiskSequence(timestamps, scores, provider="test"),
        prediction_timestamp=4.0,
        slope_window_seconds=3.0,
        peak_window_seconds=2.0,
        trigger=TriggerConfig(
            policy="risk_or_slope",
            risk_threshold=0.9,
            slope_threshold=0.09,
        ),
    )
    assert features.current_risk == pytest.approx(0.5)
    assert features.risk_slope == pytest.approx(0.1)
    assert features.recent_peak == pytest.approx(0.5)
    assert features.trigger_timestamp == pytest.approx(1.0)


def test_future_risk_cannot_enter_features() -> None:
    sequence = RiskSequence(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.1, 0.2, 0.3, 1.0], dtype=np.float32),
        provider="test",
    )
    features = extract_risk_features(
        sequence,
        prediction_timestamp=2.0,
        slope_window_seconds=2.0,
        peak_window_seconds=5.0,
    )
    assert features.current_risk == pytest.approx(0.3)
    assert features.recent_peak == pytest.approx(0.3)
    assert features.prediction_timestamp == pytest.approx(2.0)


def test_nan_warmup_is_ignored_but_infinities_are_rejected() -> None:
    sequence = RiskSequence(
        np.arange(5, dtype=np.float64),
        np.array([np.nan, np.nan, 0.2, 0.3, 0.4], dtype=np.float32),
        provider="test",
    )
    features = extract_risk_features(sequence)
    assert features.valid_observations == 3
    assert features.current_risk == pytest.approx(0.4)
    with pytest.raises(ValueError, match="infinities"):
        RiskSequence(
            np.array([0.0, 1.0]),
            np.array([0.1, np.inf], dtype=np.float32),
            provider="test",
        )


def test_no_finite_observation_fails_closed() -> None:
    sequence = RiskSequence(
        np.array([0.0, 1.0]),
        np.array([np.nan, np.nan], dtype=np.float32),
        provider="test",
    )
    with pytest.raises(ValueError, match="No finite"):
        extract_risk_features(sequence)
