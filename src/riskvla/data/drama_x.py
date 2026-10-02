"""Strict loader for canonical and access-populated DRAMA-X JSONL files."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from riskvla.data.labels import ActionMapping

REQUIRED_FIELDS = {
    "image_path",
    "video_path",
    "Risk",
    "Pedestrians",
    "Cyclists",
    "suggested_action",
    "id",
}


class DramaXError(ValueError):
    """Raised when DRAMA-X data violate the expected contract."""


@dataclass(frozen=True)
class DramaXRecord:
    sample_id: str
    native_action: str
    risk_label: str
    image_path: str
    video_path: str
    pedestrians: Mapping[str, Any]
    cyclists: Mapping[str, Any]
    extras: Mapping[str, Any]
    macro_action: str | None = None
    include_in_evaluation: bool = True

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        action_mapping: ActionMapping | None = None,
    ) -> DramaXRecord:
        missing = REQUIRED_FIELDS - set(payload)
        if missing:
            raise DramaXError(f"Missing DRAMA-X fields: {sorted(missing)}")

        sample_id = payload["id"]
        native_action = payload["suggested_action"]
        risk_label = payload["Risk"]
        image_path = payload["image_path"]
        video_path = payload["video_path"]
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise DramaXError("id must be a non-empty string")
        if not isinstance(native_action, str) or not native_action:
            raise DramaXError(f"suggested_action must be a string for {sample_id!r}")
        if risk_label not in {"Yes", "No", "N/A"}:
            raise DramaXError(f"Unexpected Risk value {risk_label!r} for {sample_id!r}")
        if not isinstance(image_path, str) or not isinstance(video_path, str):
            raise DramaXError(f"Media paths must be strings for {sample_id!r}")
        if not isinstance(payload["Pedestrians"], dict) or not isinstance(
            payload["Cyclists"], dict
        ):
            raise DramaXError(f"Actor annotations must be objects for {sample_id!r}")

        macro_action: str | None = None
        include = True
        if action_mapping is not None:
            macro_action = action_mapping.map_label(native_action)
            include = action_mapping.is_evaluable(native_action)

        return cls(
            sample_id=sample_id,
            native_action=native_action,
            risk_label=risk_label,
            image_path=image_path,
            video_path=video_path,
            pedestrians=payload["Pedestrians"],
            cyclists=payload["Cyclists"],
            extras={key: value for key, value in payload.items() if key not in REQUIRED_FIELDS},
            macro_action=macro_action,
            include_in_evaluation=include,
        )

    def source_group(self, *, explicit_field: str | None = None) -> str | None:
        """Return a defensible source group or ``None``.

        Exact video references group rows that came from the same populated
        video. For custom owner-supplied metadata, callers must name an
        explicit field. Image-ID prefix heuristics are deliberately absent.
        """
        if explicit_field is not None:
            value = self.extras.get(explicit_field)
            if isinstance(value, (str, int)) and str(value).strip():
                return f"{explicit_field}:{value}"
            return None
        if self.video_path.strip():
            return f"video:{self.video_path.strip()}"
        return None

    def to_manifest_item(self, group_id: str) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "group_id": group_id,
            "native_action": self.native_action,
            "macro_action": self.macro_action,
            "risk_label": self.risk_label,
            "image_path": self.image_path,
            "video_path": self.video_path,
        }


class DramaXDataset(Sequence[DramaXRecord]):
    def __init__(self, records: list[DramaXRecord], *, source_path: Path) -> None:
        self._records = records
        self.source_path = source_path

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        action_mapping: ActionMapping | None = None,
    ) -> DramaXDataset:
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        records: list[DramaXRecord] = []
        seen: set[str] = set()
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DramaXError(
                        f"Invalid JSON at {source_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise DramaXError(
                        f"Expected an object at {source_path}:{line_number}"
                    )
                record = DramaXRecord.from_dict(payload, action_mapping=action_mapping)
                if record.sample_id in seen:
                    raise DramaXError(f"Duplicate id {record.sample_id!r}")
                seen.add(record.sample_id)
                records.append(record)
        if not records:
            raise DramaXError(f"No records found in {source_path}")
        return cls(records, source_path=source_path)

    def __getitem__(self, index: int) -> DramaXRecord:
        return self._records[index]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[DramaXRecord]:
        return iter(self._records)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        with self.source_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def inventory(self) -> dict[str, int]:
        return dict(sorted(Counter(record.native_action for record in self).items()))

    def summary(self, *, group_field: str | None = None) -> dict[str, Any]:
        group_ids = [
            record.source_group(explicit_field=group_field) for record in self
        ]
        return {
            "records": len(self),
            "unique_ids": len({record.sample_id for record in self}),
            "sha256": self.sha256,
            "native_actions": self.inventory(),
            "risk_labels": dict(
                sorted(Counter(record.risk_label for record in self).items())
            ),
            "nonempty_image_paths": sum(bool(record.image_path.strip()) for record in self),
            "nonempty_video_paths": sum(bool(record.video_path.strip()) for record in self),
            "records_with_group": sum(group is not None for group in group_ids),
            "unique_groups": len({group for group in group_ids if group is not None}),
            "evaluable_records": sum(record.include_in_evaluation for record in self),
        }
