from __future__ import annotations

import numpy as np
import pytest

from riskvla.data.cutoffs import (
    CutoffError,
    CutoffPolicy,
    assert_causal_frames,
    candidate_positive_cutoffs,
    comparable_negative_cutoffs,
    explicit_negative_cutoff,
    select_pre_cutoff_frames,
)
from riskvla.risk.base import RiskSequence
from riskvla.risk.features import extract_risk_features


def test_positive_cutoffs_are_configurable_and_pre_event() -> None:
    policy = CutoffPolicy(positive_offsets_seconds=(3.0, 2.0, 1.0))
    points = candidate_positive_cutoffs(20.0, policy)
    assert [point.timestamp for point in points] == [17.0, 18.0, 19.0]
    assert all(point.timestamp < point.event_time for point in points)
    short = candidate_positive_cutoffs(2.5, policy)
    assert [point.offset_seconds for point in short] == [2.0, 1.0]


def test_negative_cutoffs_do_not_invent_events() -> None:
    point = explicit_negative_cutoff(18.0, duration_seconds=40.0)
    assert point.event_time is None
    assert point.is_event is False
    anchors = comparable_negative_cutoffs(40.0, [17.0, 18.0, 50.0])
    assert [item.timestamp for item in anchors] == [17.0, 18.0]
    with pytest.raises(CutoffError, match="explicit anchor"):
        comparable_negative_cutoffs(40.0, [])


def test_post_cutoff_and_post_event_frames_are_rejected() -> None:
    with pytest.raises(CutoffError, match="after observation_cutoff"):
        assert_causal_frames([1.0, 4.2], observation_cutoff=4.0, event_time=5.0)
    with pytest.raises(CutoffError, match="post-event"):
        assert_causal_frames([4.9], observation_cutoff=4.5, event_time=4.9)
    with pytest.raises(CutoffError, match="strictly before"):
        assert_causal_frames([], observation_cutoff=5.0, event_time=5.0)


def test_sampled_frames_stay_at_or_before_cutoff_and_before_event() -> None:
    timestamps = np.arange(0.0, 21.0, 0.5)
    selection = select_pre_cutoff_frames(
        timestamps,
        observation_cutoff=18.0,
        event_time=20.0,
        frame_count=4,
    )
    assert len(selection.timestamps) == 4
    assert max(selection.timestamps) <= 18.0
    assert max(selection.timestamps) < 20.0
    assert selection.timestamps == tuple(sorted(selection.timestamps))


def test_risk_features_ignore_scores_after_the_cutoff() -> None:
    sequence = RiskSequence(
        timestamps=np.array([16.0, 17.0, 18.0, 19.0, 20.0]),
        risk_scores=np.array([0.2, 0.3, 0.4, 0.95, 0.99], dtype=np.float32),
        provider="synthetic",
    )
    features = extract_risk_features(
        sequence,
        prediction_timestamp=18.0,
        slope_window_seconds=3.0,
        peak_window_seconds=3.0,
    )
    assert features.current_risk == pytest.approx(0.4)
    assert features.recent_peak == pytest.approx(0.4)
    assert features.prediction_timestamp == pytest.approx(18.0)
