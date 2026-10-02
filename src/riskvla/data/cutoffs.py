"""Configurable observation cutoffs with explicit leakage checks.

Positive cutoffs are candidates relative to a verified ``time_of_event``.
Negative cutoffs are explicit observation times. They are not synthesized
events and they are not action labels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from riskvla.vision.frame_sampler import FrameSelection, UniformFrameSampler


class CutoffError(ValueError):
    """Raised when a cutoff is invalid or would leak future evidence."""


@dataclass(frozen=True)
class CutoffPolicy:
    """Which pre-event offsets are candidates. Experiments may use a subset."""

    positive_offsets_seconds: tuple[float, ...] = (3.0, 2.0, 1.0)
    minimum_cutoff_seconds: float = 0.0
    name: str = "pre_event_offsets_v1"

    def __post_init__(self) -> None:
        if not self.positive_offsets_seconds:
            raise CutoffError("At least one positive offset is required")
        if any(
            not math.isfinite(offset) or offset <= 0 for offset in self.positive_offsets_seconds
        ):
            raise CutoffError("Positive offsets must be finite and greater than zero")
        if not math.isfinite(self.minimum_cutoff_seconds) or self.minimum_cutoff_seconds < 0:
            raise CutoffError("minimum_cutoff_seconds must be finite and non-negative")

    @classmethod
    def from_config(cls, payload: dict[str, object] | None) -> CutoffPolicy:
        if not payload:
            return cls()
        raw_offsets = payload.get("positive_offsets_seconds", (3.0, 2.0, 1.0))
        if not isinstance(raw_offsets, (list, tuple)) or not raw_offsets:
            raise CutoffError("positive_offsets_seconds must be a non-empty list")
        minimum = payload.get("minimum_cutoff_seconds", 0.0)
        name = payload.get("name", payload.get("policy", "pre_event_offsets_v1"))
        if not isinstance(name, str) or not name.strip():
            raise CutoffError("cutoff policy name must be a non-empty string")
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
            raise CutoffError("minimum_cutoff_seconds must be a number")
        return cls(
            positive_offsets_seconds=tuple(float(value) for value in raw_offsets),
            minimum_cutoff_seconds=float(minimum),
            name=name.strip(),
        )


@dataclass(frozen=True)
class ObservationPoint:
    timestamp: float
    kind: str
    event_time: float | None
    offset_seconds: float | None
    is_event: bool = False

    def __post_init__(self) -> None:
        if self.is_event:
            raise CutoffError("Observation points must not be marked as events")
        if self.kind not in {"positive_pre_event", "negative_explicit"}:
            raise CutoffError(f"Unknown observation kind {self.kind!r}")
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise CutoffError("Observation timestamp must be finite and non-negative")
        if self.kind == "positive_pre_event":
            if self.event_time is None or self.timestamp >= self.event_time:
                raise CutoffError("Positive cutoff must be strictly before time_of_event")
        elif self.event_time is not None:
            raise CutoffError("Negative observation points must not invent time_of_event")


def candidate_positive_cutoffs(
    event_time: float,
    policy: CutoffPolicy | None = None,
) -> list[ObservationPoint]:
    """Return configured pre-event cutoffs that fall inside the clip.

    Offsets that would land before ``minimum_cutoff_seconds`` are omitted.
    Callers choose which of the remaining candidates an experiment uses.
    """
    selected = policy or CutoffPolicy()
    if isinstance(event_time, bool) or not isinstance(event_time, (int, float)):
        raise CutoffError("time_of_event must be a number")
    if not math.isfinite(event_time) or event_time <= 0:
        raise CutoffError("time_of_event must be finite and positive")
    points: list[ObservationPoint] = []
    for offset in selected.positive_offsets_seconds:
        timestamp = float(event_time) - float(offset)
        if timestamp < selected.minimum_cutoff_seconds:
            continue
        if timestamp >= float(event_time):
            continue
        points.append(
            ObservationPoint(
                timestamp=timestamp,
                kind="positive_pre_event",
                event_time=float(event_time),
                offset_seconds=float(offset),
            )
        )
    return points


def explicit_negative_cutoff(
    timestamp: float,
    *,
    duration_seconds: float | None = None,
) -> ObservationPoint:
    """Store a caller-supplied observation time for a negative clip.

    The timestamp is not an event and is not a hazard label.
    """
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise CutoffError("Negative observation timestamp must be a number")
    if duration_seconds is not None:
        if (
            isinstance(duration_seconds, bool)
            or not isinstance(duration_seconds, (int, float))
            or not math.isfinite(duration_seconds)
            or duration_seconds < 0
        ):
            raise CutoffError("duration_seconds must be finite and non-negative")
        if float(timestamp) > float(duration_seconds):
            raise CutoffError("Negative cutoff exceeds the clip duration")
    return ObservationPoint(
        timestamp=float(timestamp),
        kind="negative_explicit",
        event_time=None,
        offset_seconds=None,
    )


def comparable_negative_cutoffs(
    duration_seconds: float,
    anchor_times: Sequence[float],
) -> list[ObservationPoint]:
    """Place explicit anchor times on a negative clip when they fit.

    Anchors are temporally comparable observation times supplied by the caller.
    They are dropped when they fall outside the clip. This function does not
    create a fake event, alert, or hazard.
    """
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(duration_seconds)
        or duration_seconds < 0
    ):
        raise CutoffError("duration_seconds must be finite and non-negative")
    if not anchor_times:
        raise CutoffError(
            "Negative cutoffs require explicit anchor times; none were provided"
        )
    points: list[ObservationPoint] = []
    for anchor in anchor_times:
        if isinstance(anchor, bool) or not isinstance(anchor, (int, float)):
            raise CutoffError("Anchor times must be numbers")
        if not math.isfinite(anchor) or anchor < 0:
            raise CutoffError("Anchor times must be finite and non-negative")
        if float(anchor) > float(duration_seconds):
            continue
        points.append(explicit_negative_cutoff(float(anchor), duration_seconds=duration_seconds))
    if not points:
        raise CutoffError("No anchor time falls inside the negative clip")
    return points


def assert_causal_frames(
    frame_times: Sequence[float],
    *,
    observation_cutoff: float,
    event_time: float | None = None,
) -> None:
    """Require ``frame_time <= observation_cutoff`` and, when set, ``< event``."""
    if (
        isinstance(observation_cutoff, bool)
        or not isinstance(observation_cutoff, (int, float))
        or not math.isfinite(observation_cutoff)
        or observation_cutoff < 0
    ):
        raise CutoffError("observation_cutoff must be finite and non-negative")
    if event_time is not None:
        if (
            isinstance(event_time, bool)
            or not isinstance(event_time, (int, float))
            or not math.isfinite(event_time)
            or event_time <= 0
        ):
            raise CutoffError("time_of_event must be finite and positive")
        if float(observation_cutoff) >= float(event_time):
            raise CutoffError(
                "observation_cutoff must be strictly before time_of_event"
            )
    for frame_time in frame_times:
        if isinstance(frame_time, bool) or not isinstance(frame_time, (int, float)):
            raise CutoffError("frame timestamps must be numbers")
        if not math.isfinite(frame_time):
            raise CutoffError("frame timestamps must be finite")
        if event_time is not None and float(frame_time) >= float(event_time):
            raise CutoffError("post-event frame leakage detected")
        if float(frame_time) > float(observation_cutoff) + 1e-9:
            raise CutoffError("frame_time is after observation_cutoff")


def select_pre_cutoff_frames(
    timestamps: Sequence[float] | NDArray[np.float64],
    *,
    observation_cutoff: float,
    event_time: float | None = None,
    frame_count: int = 4,
) -> FrameSelection:
    """Sample frames that are eligible at ``observation_cutoff``."""
    assert_causal_frames(
        (),
        observation_cutoff=observation_cutoff,
        event_time=event_time,
    )
    selection = UniformFrameSampler(frame_count).select(
        timestamps,
        prediction_cutoff=float(observation_cutoff),
        event_time=None if event_time is None else float(event_time),
    )
    assert_causal_frames(
        selection.timestamps,
        observation_cutoff=observation_cutoff,
        event_time=event_time,
    )
    return selection
