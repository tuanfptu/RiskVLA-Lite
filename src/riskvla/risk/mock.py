"""Deterministic synthetic risk provider for tests and UI plumbing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from riskvla.risk.base import RiskProvider, RiskSequence


class MockRiskProvider(RiskProvider):
    """Return a configured sequence and label it unambiguously as synthetic."""

    name = "mock"
    version = "synthetic-v1"

    def __init__(
        self,
        *,
        timestamps: list[float] | np.ndarray | None = None,
        risk_scores: list[float] | np.ndarray | None = None,
        require_media: bool = True,
    ) -> None:
        if timestamps is None:
            timestamps = np.arange(0.0, 4.0, 0.125, dtype=np.float64)
        timestamps_array = np.asarray(timestamps, dtype=np.float64)
        if risk_scores is None:
            midpoint = float(np.median(timestamps_array))
            risk_scores = 1.0 / (1.0 + np.exp(-3.0 * (timestamps_array - midpoint)))
        self._timestamps = timestamps_array
        self._risk_scores = np.asarray(risk_scores, dtype=np.float32)
        self.require_media = require_media

    def predict(self, video_path: str | Path) -> RiskSequence:
        path = Path(video_path)
        if self.require_media and not path.is_file():
            raise FileNotFoundError(path)
        return RiskSequence(
            timestamps=self._timestamps.copy(),
            risk_scores=self._risk_scores.copy(),
            provider=self.name,
            metadata={
                "version": self.version,
                "synthetic": True,
                "warning": "Not model output; do not use for benchmark claims.",
                "input_path": str(path),
            },
        )
