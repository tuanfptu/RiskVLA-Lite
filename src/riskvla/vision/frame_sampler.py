"""Causal frame selection with explicit event-leakage checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image


class FrameSamplingError(ValueError):
    """Raised when a valid causal frame set cannot be selected."""


@dataclass(frozen=True)
class FrameSelection:
    indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    strategy: str
    prediction_cutoff: float
    event_time: float | None

    def __post_init__(self) -> None:
        if not self.indices or len(self.indices) != len(self.timestamps):
            raise FrameSamplingError("Frame selection must contain aligned entries")
        if tuple(sorted(self.indices)) != self.indices or len(set(self.indices)) != len(
            self.indices
        ):
            raise FrameSamplingError("Selected indices must be unique and ordered")
        if tuple(sorted(self.timestamps)) != self.timestamps:
            raise FrameSamplingError("Selected timestamps must be ordered")
        if any(time > self.prediction_cutoff + 1e-9 for time in self.timestamps):
            raise FrameSamplingError("Selected frame occurs after prediction cutoff")
        if self.event_time is not None and any(
            time >= self.event_time for time in self.timestamps
        ):
            raise FrameSamplingError("Post-event frame leakage detected")


def _validate_timestamps(timestamps: NDArray[np.float64]) -> None:
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise FrameSamplingError("Frame timestamps must be a non-empty vector")
    if not np.isfinite(timestamps).all() or timestamps[0] < 0:
        raise FrameSamplingError("Frame timestamps must be finite and non-negative")
    if np.any(np.diff(timestamps) <= 0):
        raise FrameSamplingError("Frame timestamps must be strictly increasing")


def _uniform_positions(eligible_indices: NDArray[np.int64], count: int) -> tuple[int, ...]:
    if count <= 0:
        raise FrameSamplingError("Frame count must be positive")
    if eligible_indices.size < count:
        raise FrameSamplingError(
            f"Requested {count} frames but only {eligible_indices.size} are eligible"
        )
    positions = np.linspace(0, eligible_indices.size - 1, count)
    selected = eligible_indices[np.rint(positions).astype(np.int64)]
    if np.unique(selected).size != count:
        # Defensive fallback for unusual rounding/platform behavior.
        selected = eligible_indices[
            np.floor(np.arange(count) * eligible_indices.size / count).astype(np.int64)
        ]
    return tuple(int(index) for index in selected)


class UniformFrameSampler:
    def __init__(self, frame_count: int = 4) -> None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        self.frame_count = frame_count

    def select(
        self,
        timestamps: list[float] | NDArray[np.float64],
        *,
        prediction_cutoff: float | None = None,
        event_time: float | None = None,
        window_start: float = 0.0,
    ) -> FrameSelection:
        timeline = np.asarray(timestamps, dtype=np.float64)
        _validate_timestamps(timeline)
        cutoff = float(timeline[-1] if prediction_cutoff is None else prediction_cutoff)
        if not np.isfinite(cutoff) or cutoff < 0:
            raise FrameSamplingError("prediction_cutoff must be finite and non-negative")
        if event_time is not None:
            if not np.isfinite(event_time) or event_time <= 0:
                raise FrameSamplingError("event_time must be finite and positive")
            # Event time is an exclusive boundary.
            allowed_end = min(cutoff, float(np.nextafter(event_time, -np.inf)))
        else:
            allowed_end = cutoff
        eligible = np.flatnonzero(
            (timeline >= float(window_start)) & (timeline <= allowed_end)
        )
        indices = _uniform_positions(eligible, self.frame_count)
        return FrameSelection(
            indices=indices,
            timestamps=tuple(float(timeline[index]) for index in indices),
            strategy="uniform",
            prediction_cutoff=cutoff,
            event_time=event_time,
        )


class RiskCenteredFrameSampler:
    def __init__(
        self,
        frame_count: int = 4,
        *,
        pre_trigger_seconds: float = 1.5,
        post_trigger_seconds: float = 0.5,
    ) -> None:
        if pre_trigger_seconds < 0 or post_trigger_seconds < 0:
            raise ValueError("Trigger windows cannot be negative")
        if pre_trigger_seconds + post_trigger_seconds <= 0:
            raise ValueError("Risk-centered window must have positive width")
        self.uniform = UniformFrameSampler(frame_count)
        self.pre_trigger_seconds = pre_trigger_seconds
        self.post_trigger_seconds = post_trigger_seconds

    def select(
        self,
        timestamps: list[float] | NDArray[np.float64],
        *,
        trigger_timestamp: float,
        prediction_cutoff: float,
        event_time: float | None = None,
    ) -> FrameSelection:
        if not np.isfinite(trigger_timestamp) or trigger_timestamp < 0:
            raise FrameSamplingError("trigger_timestamp must be finite and non-negative")
        if trigger_timestamp > prediction_cutoff:
            raise FrameSamplingError("Trigger cannot occur after prediction cutoff")
        timeline = np.asarray(timestamps, dtype=np.float64)
        window_start = max(0.0, trigger_timestamp - self.pre_trigger_seconds)
        window_end = min(
            prediction_cutoff,
            trigger_timestamp + self.post_trigger_seconds,
        )
        selection = self.uniform.select(
            timeline,
            prediction_cutoff=window_end,
            event_time=event_time,
            window_start=window_start,
        )
        return FrameSelection(
            indices=selection.indices,
            timestamps=selection.timestamps,
            strategy="risk_centered",
            prediction_cutoff=prediction_cutoff,
            event_time=event_time,
        )


def decode_selected_frames(
    video_path: str | Path,
    selection: FrameSelection,
) -> list[Image.Image]:
    """Decode exact frame indices with OpenCV and return RGB PIL images."""
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for video decoding") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FrameSamplingError(f"OpenCV could not open {path}")
    wanted = set(selection.indices)
    frames: dict[int, Image.Image] = {}
    index = 0
    try:
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[index] = Image.fromarray(rgb)
                wanted.remove(index)
            index += 1
    finally:
        capture.release()
    missing = sorted(wanted)
    if missing:
        raise FrameSamplingError(f"Could not decode selected frame indices: {missing}")
    return [frames[index] for index in selection.indices]


def probe_video_timestamps(video_path: str | Path) -> NDArray[np.float64]:
    """Read frame count/FPS and construct the source-frame timeline."""
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for video probing") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FrameSamplingError(f"OpenCV could not open {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0 or count <= 0:
        raise FrameSamplingError(f"Invalid video metadata: fps={fps}, frames={count}")
    return np.arange(count, dtype=np.float64) / fps
