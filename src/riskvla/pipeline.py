"""Composable risk-conditioned action pipeline used by scripts and tests."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from riskvla.risk.base import RiskProvider
from riskvla.risk.features import RiskFeatures, TriggerConfig, extract_risk_features
from riskvla.vla.action_parser import ActionPrediction
from riskvla.vla.conditioning import ConditioningVariant


class SelectorResult(Protocol):
    prediction: ActionPrediction | None
    raw_output: str
    valid: bool
    invalid_reason: str | None
    latency_seconds: float
    peak_vram_bytes: int | None
    metadata: dict[str, Any]


class ActionSelector(Protocol):
    def predict(
        self,
        frames: Iterable[str | Path | Image.Image],
        *,
        variant: ConditioningVariant | str,
        risk_features: RiskFeatures | None = None,
        diagnostic: bool = False,
    ) -> SelectorResult: ...


@dataclass(frozen=True)
class PipelineResult:
    variant: str
    prediction: ActionPrediction | None
    valid: bool
    invalid_reason: str | None
    risk_features: RiskFeatures | None
    risk_latency_seconds: float | None
    vla_latency_seconds: float
    end_to_end_latency_seconds: float
    peak_vram_bytes: int | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prediction"] = (
            self.prediction.to_dict() if self.prediction is not None else None
        )
        payload["risk_features"] = (
            self.risk_features.to_dict() if self.risk_features is not None else None
        )
        return payload


class RiskVLAPipeline:
    def __init__(
        self,
        *,
        selector: ActionSelector,
        variant: ConditioningVariant | str,
        risk_provider: RiskProvider | None = None,
        trigger: TriggerConfig | None = None,
        slope_window_seconds: float = 2.0,
        peak_window_seconds: float = 2.0,
    ) -> None:
        self.selector = selector
        self.variant = ConditioningVariant(variant)
        self.risk_provider = risk_provider
        self.trigger = trigger or TriggerConfig()
        self.slope_window_seconds = slope_window_seconds
        self.peak_window_seconds = peak_window_seconds
        if self.variant is not ConditioningVariant.VISUAL_ONLY and risk_provider is None:
            raise ValueError(f"{self.variant.value} requires a RiskProvider")
        if self.variant is ConditioningVariant.VISUAL_ONLY and risk_provider is not None:
            raise ValueError("Visual-only pipeline must not receive a RiskProvider")

    def predict(
        self,
        *,
        frames: Iterable[str | Path | Image.Image],
        video_path: str | Path | None = None,
        prediction_timestamp: float | None = None,
        diagnostic: bool = False,
    ) -> PipelineResult:
        started = time.perf_counter()
        features: RiskFeatures | None = None
        risk_latency: float | None = None
        risk_metadata: dict[str, Any] | None = None
        if self.risk_provider is not None:
            if video_path is None:
                raise ValueError("Risk-conditioned prediction requires video_path")
            risk_started = time.perf_counter()
            sequence = self.risk_provider.predict(video_path)
            risk_latency = time.perf_counter() - risk_started
            features = extract_risk_features(
                sequence,
                prediction_timestamp=prediction_timestamp,
                slope_window_seconds=self.slope_window_seconds,
                peak_window_seconds=self.peak_window_seconds,
                trigger=self.trigger,
            )
            risk_metadata = sequence.metadata

        selector_result = self.selector.predict(
            frames,
            variant=self.variant,
            risk_features=features,
            diagnostic=diagnostic,
        )
        return PipelineResult(
            variant=self.variant.value,
            prediction=selector_result.prediction,
            valid=selector_result.valid,
            invalid_reason=selector_result.invalid_reason,
            risk_features=features,
            risk_latency_seconds=risk_latency,
            vla_latency_seconds=selector_result.latency_seconds,
            end_to_end_latency_seconds=time.perf_counter() - started,
            peak_vram_bytes=selector_result.peak_vram_bytes,
            metadata={
                "selector": selector_result.metadata,
                "risk": risk_metadata,
            },
        )
