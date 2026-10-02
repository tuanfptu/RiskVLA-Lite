from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from riskvla.pipeline import RiskVLAPipeline
from riskvla.risk.features import RiskFeatures
from riskvla.risk.mock import MockRiskProvider
from riskvla.vla.action_parser import ActionPrediction
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


class FakeSelector:
    def __init__(self) -> None:
        self.received_features: RiskFeatures | None = None

    def predict(
        self,
        frames: Iterable[Any],
        *,
        variant: ConditioningVariant | str,
        risk_features: RiskFeatures | None = None,
        diagnostic: bool = False,
    ) -> FakeSelectorResult:
        del diagnostic
        assert list(frames)
        assert ConditioningVariant(variant) is ConditioningVariant.VISUAL_PLUS_TEMPORAL_RISK
        assert risk_features is not None
        self.received_features = risk_features
        raw = '{"action":"SLOW","confidence":0.8}'
        return FakeSelectorResult(
            prediction=ActionPrediction(
                action="SLOW", confidence=0.8, raw_output=raw
            ),
            raw_output=raw,
            valid=True,
            invalid_reason=None,
            latency_seconds=0.001,
            peak_vram_bytes=None,
            metadata={"model": "fake-test-selector"},
        )


def test_mock_risk_to_action_pipeline(tmp_path: Path) -> None:
    media = tmp_path / "fixture.mp4"
    media.write_bytes(b"unit-test-placeholder")
    provider = MockRiskProvider(
        timestamps=[0.0, 1.0, 2.0, 3.0],
        risk_scores=[0.1, 0.3, 0.6, 0.9],
    )
    selector = FakeSelector()
    pipeline = RiskVLAPipeline(
        selector=selector,
        variant=ConditioningVariant.VISUAL_PLUS_TEMPORAL_RISK,
        risk_provider=provider,
    )
    result = pipeline.predict(
        frames=[Image.new("RGB", (8, 8), "black")],
        video_path=media,
        prediction_timestamp=2.0,
    )
    assert result.valid
    assert result.prediction is not None
    assert result.prediction.action == "SLOW"
    assert result.risk_features is not None
    assert result.risk_features.current_risk == pytest.approx(0.6)
    assert result.metadata["risk"]["synthetic"] is True
    assert selector.received_features == result.risk_features


def test_visual_only_pipeline_rejects_risk_provider() -> None:
    provider = MockRiskProvider(require_media=False)
    try:
        RiskVLAPipeline(
            selector=FakeSelector(),
            variant=ConditioningVariant.VISUAL_ONLY,
            risk_provider=provider,
        )
    except ValueError as exc:
        assert "must not receive" in str(exc)
    else:
        raise AssertionError("Visual-only pipeline accepted a risk provider")
