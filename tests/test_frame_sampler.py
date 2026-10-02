from __future__ import annotations

import numpy as np
import pytest

from riskvla.vision.frame_sampler import (
    FrameSamplingError,
    RiskCenteredFrameSampler,
    UniformFrameSampler,
)


def test_uniform_sampler_is_deterministic_and_ordered() -> None:
    timestamps = np.arange(10, dtype=np.float64) * 0.5
    selection = UniformFrameSampler(4).select(
        timestamps, prediction_cutoff=4.0
    )
    assert selection.indices == (0, 3, 5, 8)
    assert selection.timestamps == (0.0, 1.5, 2.5, 4.0)


def test_event_boundary_is_exclusive() -> None:
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    selection = UniformFrameSampler(3).select(
        timestamps,
        prediction_cutoff=2.0,
        event_time=1.5,
    )
    assert max(selection.timestamps) < 1.5
    assert 1.5 not in selection.timestamps


def test_sampler_rejects_insufficient_pre_event_frames() -> None:
    with pytest.raises(FrameSamplingError, match="only 2"):
        UniformFrameSampler(3).select(
            np.array([0.0, 1.0, 2.0]),
            prediction_cutoff=2.0,
            event_time=1.5,
        )


def test_risk_centered_sampler_never_uses_future_or_post_event_frames() -> None:
    timestamps = np.arange(0.0, 5.0, 0.25)
    selection = RiskCenteredFrameSampler(
        4, pre_trigger_seconds=1.0, post_trigger_seconds=0.5
    ).select(
        timestamps,
        trigger_timestamp=3.0,
        prediction_cutoff=3.25,
        event_time=3.5,
    )
    assert selection.strategy == "risk_centered"
    assert min(selection.timestamps) >= 2.0
    assert max(selection.timestamps) <= 3.25
    assert max(selection.timestamps) < 3.5


def test_trigger_after_prediction_is_rejected() -> None:
    with pytest.raises(FrameSamplingError, match="after prediction"):
        RiskCenteredFrameSampler().select(
            np.arange(0.0, 3.0, 0.25),
            trigger_timestamp=2.5,
            prediction_cutoff=2.0,
        )
