from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from riskvla.data.cutoffs import candidate_positive_cutoffs, select_pre_cutoff_frames
from riskvla.data.nexar import NexarRecord
from riskvla.pipeline import RiskVLAPipeline
from riskvla.risk.features import RiskFeatures
from riskvla.risk.mock import MockRiskProvider
from riskvla.vla.action_parser import ActionPrediction, parse_action_output
from riskvla.vla.conditioning import ConditioningVariant


@dataclass
class FakeSelectorResult:
    prediction: ActionPrediction | None
    raw_output: str
    valid: bool
    invalid_reason: str | None
    latency_seconds: float
    peak_vram_bytes: int | None
    metadata: dict[str, Any]


class RecordingSelector:
    def __init__(self) -> None:
        self.frame_count = 0
        self.features: RiskFeatures | None = None

    def predict(
        self,
        frames: Iterable[Any],
        *,
        variant: ConditioningVariant | str,
        risk_features: RiskFeatures | None = None,
        diagnostic: bool = False,
    ) -> FakeSelectorResult:
        del diagnostic
        self.frame_count = len(list(frames))
        self.features = risk_features
        raw = '{"action":"SLOW","confidence":0.84}'
        parsed = parse_action_output(raw)
        return FakeSelectorResult(
            prediction=parsed,
            raw_output=raw,
            valid=True,
            invalid_reason=None,
            latency_seconds=0.01,
            peak_vram_bytes=None,
            metadata={"model": "fake-nexar-selector", "variant": str(variant)},
        )


def test_mock_nexar_pipeline_uses_only_pre_cutoff_evidence(tmp_path: Path) -> None:
    record = NexarRecord.from_dict(
        {
            "video_id": "00010",
            "source_split": "train",
            "label": 1,
            "video_path": "train/positive/00010.mp4",
            "time_of_event": 20.0,
            "time_of_alert": 18.5,
            "scene": "Urban",
        }
    )
    cutoff = candidate_positive_cutoffs(record.time_of_event or 0.0)[1]
    assert cutoff.timestamp == pytest.approx(18.0)
    timestamps = np.arange(0.0, 21.0, 0.5)
    selection = select_pre_cutoff_frames(
        timestamps,
        observation_cutoff=cutoff.timestamp,
        event_time=record.time_of_event,
        frame_count=4,
    )
    assert max(selection.timestamps) <= cutoff.timestamp
    assert record.time_of_event is not None
    assert max(selection.timestamps) < record.time_of_event

    media = tmp_path / "00010.mp4"
    media.write_bytes(b"not-a-real-video")
    provider = MockRiskProvider(
        timestamps=[16.0, 17.0, 18.0, 19.0, 20.0],
        risk_scores=[0.15, 0.25, 0.35, 0.9, 0.99],
    )
    selector = RecordingSelector()
    pipeline = RiskVLAPipeline(
        selector=selector,
        variant=ConditioningVariant.VISUAL_PLUS_TEMPORAL_RISK,
        risk_provider=provider,
    )
    result = pipeline.predict(
        frames=[Image.new("RGB", (8, 8), "black") for _ in selection.timestamps],
        video_path=media,
        prediction_timestamp=cutoff.timestamp,
    )
    assert result.valid
    assert result.prediction is not None
    assert result.prediction.action == "SLOW"
    assert result.prediction.confidence == pytest.approx(0.84)
    assert selector.frame_count == 4
    assert result.risk_features is not None
    assert result.risk_features.current_risk == pytest.approx(0.35)
    assert result.risk_features.recent_peak == pytest.approx(0.35)
    with pytest.raises(Exception, match="Unknown action"):
        parse_action_output('{"action":"MANEUVER","confidence":0.5}')
