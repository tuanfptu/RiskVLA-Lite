"""Human action annotations for the Nexar subset.

This module validates manually written records. It does not infer actions from
collision labels, BADAS scores, alert times, or model output.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from riskvla.constants import ACTIONS
from riskvla.data.nexar import NexarRecord
from riskvla.data.splits import GroupedSample

EGO_RELEVANCE = ("none", "indirect", "direct")
ANNOTATION_FIELDS = (
    "video_id",
    "source_split",
    "collision_label",
    "time_of_event",
    "time_of_alert",
    "observation_cutoff",
    "action",
    "actionable_from",
    "ego_relevance",
    "confidence",
    "ambiguous",
    "annotator",
    "notes",
)


class AnnotationError(ValueError):
    """Raised when a human action annotation violates the schema."""


def _finite_number(value: Any, *, field: str, video_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnnotationError(f"{field} for {video_id!r} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise AnnotationError(f"{field} for {video_id!r} must be finite")
    return number


def _optional_time(value: Any, *, field: str, video_id: str) -> float | None:
    if value is None:
        return None
    number = _finite_number(value, field=field, video_id=video_id)
    if number < 0:
        raise AnnotationError(f"{field} for {video_id!r} must be non-negative")
    return number


@dataclass(frozen=True)
class ActionAnnotation:
    video_id: str
    source_split: str
    collision_label: int
    time_of_event: float | None
    time_of_alert: float | None
    observation_cutoff: float
    action: str
    actionable_from: float | None
    ego_relevance: str
    confidence: int
    ambiguous: bool
    annotator: str
    notes: str
    nexar_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.nexar_metadata is not None:
            object.__setattr__(self, "nexar_metadata", dict(self.nexar_metadata))

    @property
    def sample_id(self) -> str:
        return f"{self.video_id}@{self.observation_cutoff:.3f}"

    @property
    def group_id(self) -> str:
        return self.video_id

    def to_grouped_sample(self) -> GroupedSample:
        """Group every cutoff from one source video into the same split."""
        return GroupedSample(
            sample_id=self.sample_id,
            group_id=self.group_id,
            label=self.action,
            attributes={
                "video_id": self.video_id,
                "source_split": self.source_split,
                "collision_label": self.collision_label,
                "observation_cutoff": self.observation_cutoff,
                "action": self.action,
            },
        )


def validate_annotation(payload: Mapping[str, Any]) -> ActionAnnotation:
    """Validate one annotation. Metadata nested under ``nexar_metadata`` is preserved only."""
    if not isinstance(payload, Mapping):
        raise AnnotationError("Annotation must be an object")
    missing = [field for field in ANNOTATION_FIELDS if field not in payload]
    if missing:
        raise AnnotationError(f"Missing annotation fields: {missing}")
    unknown = sorted(set(payload) - set(ANNOTATION_FIELDS) - {"nexar_metadata"})
    if unknown:
        raise AnnotationError(f"Unexpected annotation fields: {unknown}")

    video_id = payload["video_id"]
    if not isinstance(video_id, str) or not video_id.strip():
        raise AnnotationError("video_id must be a non-empty string")
    video_id = video_id.strip()
    source_split = payload["source_split"]
    if not isinstance(source_split, str) or not source_split.strip():
        raise AnnotationError(f"source_split must be a non-empty string for {video_id!r}")

    collision_label = payload["collision_label"]
    if isinstance(collision_label, bool) or collision_label not in (0, 1):
        raise AnnotationError(f"collision_label for {video_id!r} must be 0 or 1")

    time_of_event = _optional_time(
        payload["time_of_event"], field="time_of_event", video_id=video_id
    )
    time_of_alert = _optional_time(
        payload["time_of_alert"], field="time_of_alert", video_id=video_id
    )
    if collision_label == 1 and time_of_event is None:
        raise AnnotationError(f"Positive annotation {video_id!r} requires time_of_event")
    if collision_label == 0 and time_of_event is not None:
        raise AnnotationError(
            f"Negative annotation {video_id!r} must not invent time_of_event"
        )

    observation_cutoff = _finite_number(
        payload["observation_cutoff"], field="observation_cutoff", video_id=video_id
    )
    if observation_cutoff < 0:
        raise AnnotationError(f"observation_cutoff for {video_id!r} must be non-negative")
    if time_of_event is not None and observation_cutoff >= time_of_event:
        raise AnnotationError(
            f"observation_cutoff for {video_id!r} must be strictly before time_of_event"
        )

    action = payload["action"]
    if action not in ACTIONS:
        raise AnnotationError(
            f"Unknown action {action!r} for {video_id!r}; expected one of {ACTIONS}"
        )

    actionable_from = _optional_time(
        payload["actionable_from"], field="actionable_from", video_id=video_id
    )
    if actionable_from is not None and actionable_from > observation_cutoff:
        raise AnnotationError(
            f"actionable_from for {video_id!r} cannot be after observation_cutoff"
        )
    if (
        actionable_from is not None
        and time_of_event is not None
        and actionable_from >= time_of_event
    ):
        raise AnnotationError(
            f"actionable_from for {video_id!r} must be strictly before time_of_event"
        )

    ego_relevance = payload["ego_relevance"]
    if ego_relevance not in EGO_RELEVANCE:
        raise AnnotationError(
            f"ego_relevance for {video_id!r} must be one of {EGO_RELEVANCE}"
        )
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 1 <= confidence <= 5:
        raise AnnotationError(f"confidence for {video_id!r} must be an integer from 1 to 5")
    ambiguous = payload["ambiguous"]
    if not isinstance(ambiguous, bool):
        raise AnnotationError(f"ambiguous for {video_id!r} must be boolean")
    annotator = payload["annotator"]
    notes = payload["notes"]
    if not isinstance(annotator, str) or not annotator.strip():
        raise AnnotationError(f"annotator for {video_id!r} must be a non-empty string")
    if not isinstance(notes, str):
        raise AnnotationError(f"notes for {video_id!r} must be a string")

    metadata = payload.get("nexar_metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise AnnotationError(f"nexar_metadata for {video_id!r} must be an object")

    return ActionAnnotation(
        video_id=video_id,
        source_split=source_split.strip(),
        collision_label=int(collision_label),
        time_of_event=time_of_event,
        time_of_alert=time_of_alert,
        observation_cutoff=observation_cutoff,
        action=action,
        actionable_from=actionable_from,
        ego_relevance=ego_relevance,
        confidence=confidence,
        ambiguous=ambiguous,
        annotator=annotator.strip(),
        notes=notes,
        nexar_metadata=metadata,
    )


def load_annotations(path: str | Path) -> list[ActionAnnotation]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    records: list[ActionAnnotation] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnnotationError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            record = validate_annotation(payload)
            if record.sample_id in seen:
                raise AnnotationError(f"Duplicate annotation {record.sample_id!r}")
            seen.add(record.sample_id)
            records.append(record)
    return records


def attach_preserved_metadata(
    annotation: ActionAnnotation,
    record: NexarRecord,
) -> ActionAnnotation:
    """Copy official Nexar metadata beside an annotation without changing its action."""
    if annotation.video_id != record.video_id:
        raise AnnotationError(
            f"Cannot attach metadata for {record.video_id!r} to annotation {annotation.video_id!r}"
        )
    metadata = record.preserved_metadata()
    metadata.pop("action", None)
    return ActionAnnotation(
        video_id=annotation.video_id,
        source_split=annotation.source_split,
        collision_label=annotation.collision_label,
        time_of_event=annotation.time_of_event,
        time_of_alert=annotation.time_of_alert,
        observation_cutoff=annotation.observation_cutoff,
        action=annotation.action,
        actionable_from=annotation.actionable_from,
        ego_relevance=annotation.ego_relevance,
        confidence=annotation.confidence,
        ambiguous=annotation.ambiguous,
        annotator=annotation.annotator,
        notes=annotation.notes,
        nexar_metadata=metadata,
    )
