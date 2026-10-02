"""Normalize official Nexar collision-prediction records.

Media files are never downloaded by this module. Tests use synthetic records
and tiny CSV fixtures. Paths are references, not assumed local files.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_ID = "nexar-ai/nexar_collision_prediction"
DATASET_REVISION = "7535d0656dac31d7da2846913bb69c6331d2a70a"
DATASET_LAST_MODIFIED = "2026-09-29T14:37:45.000Z"
DATASET_CREATED_AT = "2025-01-28T15:42:33.000Z"
USED_STORAGE_BYTES = 31_379_211_365
CARD_TOTAL_FILE_SIZE = "31.4 GB"
OFFICIAL_SPLITS = ("train", "test-public", "test-private")

# Counts from the public Hugging Face file listing at DATASET_REVISION.
# Sibling sizes are omitted for this gated repository, so these are file
# counts rather than byte sizes.
VERIFIED_VIDEO_COUNTS: dict[str, int] = {
    "train/positive": 750,
    "train/negative": 750,
    "test-public/positive": 334,
    "test-public/negative": 333,
    "test-private/positive": 338,
    "test-private/negative": 339,
}

_NULL_TOKENS = {"", "none", "null", "nan"}
_TIME_FIELDS = ("time_of_event", "time_of_alert", "time_to_accident")


class NexarError(ValueError):
    """Raised when a Nexar record violates the verified dataset contract."""


def _video_stem(reference: str) -> str:
    name = Path(reference.replace("\\", "/")).name
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    stem = name.strip()
    if not stem:
        raise NexarError("video reference does not contain an id")
    return stem


def _optional_float(value: Any, *, field: str, video_id: str) -> float | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in _NULL_TOKENS:
            return None
        try:
            value = float(stripped)
        except ValueError as exc:
            raise NexarError(
                f"{field} for {video_id!r} is not a number: {stripped!r}"
            ) from exc
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NexarError(f"{field} for {video_id!r} must be a number or null")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise NexarError(f"{field} for {video_id!r} must be finite and non-negative")
    return number


def _optional_text(value: Any, *, field: str, video_id: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise NexarError(f"{field} for {video_id!r} must be a string or null")
    stripped = value.strip()
    return stripped or None


def _collision_label(value: Any, *, video_id: str) -> int:
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        value = int(value.strip())
    if isinstance(value, bool) or value not in (0, 1):
        raise NexarError(f"collision label for {video_id!r} must be 0 or 1")
    return int(value)


@dataclass(frozen=True)
class NexarRecord:
    """One official clip, independent of any safety-action label."""

    video_id: str
    source_split: str
    collision_label: int
    video_path: str
    time_of_event: float | None
    time_of_alert: float | None
    time_to_accident: float | None
    light_conditions: str | None
    weather: str | None
    scene: str | None
    original_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.source_split not in OFFICIAL_SPLITS:
            raise NexarError(f"Unknown Nexar split {self.source_split!r}")
        if self.collision_label not in (0, 1):
            raise NexarError("collision_label must be 0 or 1")
        if self.collision_label == 1 and self.time_of_event is None:
            raise NexarError(
                f"Positive clip {self.video_id!r} requires time_of_event"
            )
        if self.collision_label == 0 and self.time_of_event is not None:
            raise NexarError(
                f"Negative clip {self.video_id!r} must not carry time_of_event"
            )
        object.__setattr__(self, "original_metadata", dict(self.original_metadata))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        source_split: str | None = None,
        collision_label: int | None = None,
        video_root: str | None = None,
    ) -> NexarRecord:
        if not isinstance(payload, Mapping):
            raise NexarError("Nexar record must be an object")
        reference = payload.get("video_path", payload.get("video", payload.get("file_name")))
        if not isinstance(reference, str) or not reference.strip():
            raise NexarError("Nexar record requires video, video_path, or file_name")
        video_id = payload.get("video_id") or payload.get("id") or _video_stem(reference)
        if not isinstance(video_id, str) or not video_id.strip():
            raise NexarError("video_id must be a non-empty string")
        video_id = _video_stem(video_id) if video_id.lower().endswith(".mp4") else video_id.strip()

        split = source_split if source_split is not None else payload.get("source_split")
        if not isinstance(split, str) or not split.strip():
            raise NexarError(f"source_split is required for {video_id!r}")

        if collision_label is None:
            raw_label = payload.get("collision_label", payload.get("label"))
        else:
            raw_label = collision_label
        label = _collision_label(raw_label, video_id=video_id)

        video_path = reference.strip()
        if video_root and not video_path.startswith(("/", "train/", "test-")):
            polarity = "positive" if label == 1 else "negative"
            video_path = str(Path(video_root) / split / polarity / Path(video_path).name)
        elif "/" not in video_path.replace("\\", "/"):
            polarity = "positive" if label == 1 else "negative"
            prefix = f"{video_root}/" if video_root else ""
            video_path = f"{prefix}{split}/{polarity}/{Path(video_path).name}"

        times = {
            field: _optional_float(payload.get(field), field=field, video_id=video_id)
            for field in _TIME_FIELDS
        }
        consumed = {
            "video_id",
            "id",
            "source_split",
            "collision_label",
            "label",
            "video",
            "video_path",
            "file_name",
            *_TIME_FIELDS,
            "light_conditions",
            "weather",
            "scene",
        }
        return cls(
            video_id=video_id,
            source_split=split.strip(),
            collision_label=label,
            video_path=video_path,
            time_of_event=times["time_of_event"],
            time_of_alert=times["time_of_alert"],
            time_to_accident=times["time_to_accident"],
            light_conditions=_optional_text(
                payload.get("light_conditions"), field="light_conditions", video_id=video_id
            ),
            weather=_optional_text(payload.get("weather"), field="weather", video_id=video_id),
            scene=_optional_text(payload.get("scene"), field="scene", video_id=video_id),
            original_metadata={
                key: value for key, value in payload.items() if key not in consumed
            },
        )

    def preserved_metadata(self) -> dict[str, Any]:
        """Official Nexar fields kept separate from any human action label."""
        return {
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "video_id": self.video_id,
            "source_split": self.source_split,
            "collision_label": self.collision_label,
            "video_path": self.video_path,
            "time_of_event": self.time_of_event,
            "time_of_alert": self.time_of_alert,
            "time_to_accident": self.time_to_accident,
            "light_conditions": self.light_conditions,
            "weather": self.weather,
            "scene": self.scene,
            "original_metadata": dict(self.original_metadata),
        }


class NexarDataset(Sequence[NexarRecord]):
    def __init__(self, records: list[NexarRecord], *, source: str) -> None:
        self._records = records
        self.source = source

    @classmethod
    def from_records(
        cls,
        payloads: Sequence[Mapping[str, Any]],
        *,
        source: str,
        source_split: str | None = None,
        collision_label: int | None = None,
    ) -> NexarDataset:
        records: list[NexarRecord] = []
        seen: set[str] = set()
        for index, payload in enumerate(payloads):
            record = NexarRecord.from_dict(
                payload,
                source_split=source_split,
                collision_label=collision_label,
            )
            if record.video_id in seen:
                raise NexarError(f"Duplicate video id {record.video_id!r} at index {index}")
            seen.add(record.video_id)
            records.append(record)
        if not records:
            raise NexarError(f"No Nexar records found in {source}")
        return cls(records, source=source)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        source_split: str | None = None,
    ) -> NexarDataset:
        source_path = Path(path).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        payloads: list[dict[str, Any]] = []
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise NexarError(
                        f"Invalid JSON at {source_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise NexarError(f"Expected an object at {source_path}:{line_number}")
                payloads.append(payload)
        return cls.from_records(payloads, source=str(source_path), source_split=source_split)

    @classmethod
    def from_metadata_csv(
        cls,
        path: str | Path,
        *,
        source_split: str,
        collision_label: int,
        video_root: str | None = None,
    ) -> NexarDataset:
        """Load one official ``metadata.csv`` without opening video files."""
        source_path = Path(path).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if source_split not in OFFICIAL_SPLITS:
            raise NexarError(f"Unknown Nexar split {source_split!r}")
        label = _collision_label(collision_label, video_id=source_split)
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "file_name" not in reader.fieldnames:
                raise NexarError(f"{source_path} must include a file_name column")
            payloads = [dict(row) for row in reader]
        records: list[NexarRecord] = []
        seen: set[str] = set()
        for payload in payloads:
            record = NexarRecord.from_dict(
                payload,
                source_split=source_split,
                collision_label=label,
                video_root=video_root,
            )
            if record.video_id in seen:
                raise NexarError(f"Duplicate video id {record.video_id!r} in {source_path}")
            seen.add(record.video_id)
            records.append(record)
        if not records:
            raise NexarError(f"No rows found in {source_path}")
        return cls(records, source=str(source_path))

    def __getitem__(self, index: int) -> NexarRecord:
        return self._records[index]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[NexarRecord]:
        return iter(self._records)

    def summary(self) -> dict[str, Any]:
        return {
            "records": len(self),
            "source": self.source,
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "collision_labels": dict(
                sorted(Counter(record.collision_label for record in self).items())
            ),
            "source_splits": dict(
                sorted(Counter(record.source_split for record in self).items())
            ),
            "scenes": dict(
                sorted(Counter(record.scene or "unknown" for record in self).items())
            ),
            "with_time_of_event": sum(record.time_of_event is not None for record in self),
            "with_time_of_alert": sum(record.time_of_alert is not None for record in self),
            "with_time_to_accident": sum(
                record.time_to_accident is not None for record in self
            ),
        }
