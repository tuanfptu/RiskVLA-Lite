"""Generic risk sequence contract, independent of BADAS internals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RiskSequence:
    timestamps: NDArray[np.float64]
    risk_scores: NDArray[np.float32]
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        scores = np.asarray(self.risk_scores, dtype=np.float32)
        if timestamps.ndim != 1 or scores.ndim != 1:
            raise ValueError("timestamps and risk_scores must be one-dimensional")
        if timestamps.size == 0:
            raise ValueError("RiskSequence cannot be empty")
        if timestamps.shape != scores.shape:
            raise ValueError("timestamps and risk_scores must have equal length")
        if not np.isfinite(timestamps).all():
            raise ValueError("timestamps must all be finite")
        if timestamps[0] < 0 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps must be non-negative and strictly increasing")
        if np.isinf(scores).any():
            raise ValueError("risk_scores may contain NaN warm-up values, never infinities")
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size and (
            np.any(finite_scores < 0.0) or np.any(finite_scores > 1.0)
        ):
            raise ValueError("finite risk scores must be within [0, 1]")

        timestamps.setflags(write=False)
        scores.setflags(write=False)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "risk_scores", scores)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        return np.isfinite(self.risk_scores)

    @property
    def valid_count(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def duration_seconds(self) -> float:
        return float(self.timestamps[-1])

    def through(self, cutoff_seconds: float) -> RiskSequence:
        """Return values at or before a prediction cutoff, preventing future use."""
        if not np.isfinite(cutoff_seconds) or cutoff_seconds < 0:
            raise ValueError("cutoff_seconds must be finite and non-negative")
        mask = self.timestamps <= cutoff_seconds
        if not mask.any():
            raise ValueError("No risk observations exist at or before the cutoff")
        return RiskSequence(
            timestamps=self.timestamps[mask],
            risk_scores=self.risk_scores[mask],
            provider=self.provider,
            metadata={**self.metadata, "cutoff_seconds": cutoff_seconds},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "timestamps": self.timestamps.tolist(),
            "risk_scores": [
                None if np.isnan(score) else float(score) for score in self.risk_scores
            ],
            "metadata": self.metadata,
        }


class RiskProvider(ABC):
    """Interface implemented by frozen and synthetic risk providers."""

    name: str
    version: str

    @abstractmethod
    def predict(self, video_path: str | Path) -> RiskSequence:
        """Produce a normalized risk timeline for one video."""

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}
