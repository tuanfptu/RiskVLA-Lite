"""Frozen native-to-macro action mapping."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from riskvla.constants import ACTIONS


class ActionMappingError(ValueError):
    """Raised when action mapping data are missing or inconsistent."""


@dataclass(frozen=True)
class MappingEntry:
    native_label: str
    macro_action: str | None
    rationale: str
    include_in_evaluation: bool


@dataclass(frozen=True)
class ActionMapping:
    version: str
    status: str
    frozen_before_test_evaluation: bool
    entries: dict[str, MappingEntry]
    severity: dict[str, int]
    source: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> ActionMapping:
        mapping_path = Path(path)
        with mapping_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ActionMappingError("Mapping root must be a YAML mapping")

        declared_actions = tuple(raw.get("macro_actions", ()))
        if set(declared_actions) != set(ACTIONS) or len(declared_actions) != len(ACTIONS):
            raise ActionMappingError(
                f"macro_actions must contain each supported action exactly once: {ACTIONS}"
            )

        raw_entries = raw.get("mappings")
        if not isinstance(raw_entries, dict) or not raw_entries:
            raise ActionMappingError("mappings must be a non-empty mapping")

        entries: dict[str, MappingEntry] = {}
        for native_label, payload in raw_entries.items():
            if not isinstance(native_label, str) or not isinstance(payload, dict):
                raise ActionMappingError("Each native mapping must be a mapping keyed by a string")
            macro = payload.get("macro_action")
            include = payload.get("include_in_evaluation")
            rationale = payload.get("rationale")
            if macro is not None and macro not in ACTIONS:
                raise ActionMappingError(f"Unknown macro action {macro!r} for {native_label!r}")
            if include is True and macro is None:
                raise ActionMappingError(f"Included label {native_label!r} requires a macro action")
            if not isinstance(include, bool):
                raise ActionMappingError(
                    f"include_in_evaluation must be boolean for {native_label!r}"
                )
            if not isinstance(rationale, str) or not rationale.strip():
                raise ActionMappingError(f"Missing rationale for {native_label!r}")
            entries[native_label] = MappingEntry(
                native_label=native_label,
                macro_action=macro,
                rationale=rationale.strip(),
                include_in_evaluation=include,
            )

        severity = raw.get("severity")
        if not isinstance(severity, dict) or set(severity) != set(ACTIONS):
            raise ActionMappingError("severity must define every macro action")
        normalized_severity: dict[str, int] = {}
        for action, value in severity.items():
            if not isinstance(value, int) or value < 0:
                raise ActionMappingError(f"Invalid severity for {action!r}: {value!r}")
            normalized_severity[action] = value

        return cls(
            version=str(raw.get("version", "")),
            status=str(raw.get("status", "")),
            frozen_before_test_evaluation=bool(
                raw.get("frozen_before_test_evaluation", False)
            ),
            entries=entries,
            severity=normalized_severity,
            source=dict(raw.get("source", {})),
        )

    def require_frozen(self) -> None:
        if self.status != "frozen" or not self.frozen_before_test_evaluation:
            raise ActionMappingError(
                "Action mapping must be frozen before benchmark evaluation"
            )

    def map_label(self, native_label: str) -> str | None:
        try:
            entry = self.entries[native_label]
        except KeyError as exc:
            raise ActionMappingError(f"Unknown native action label: {native_label!r}") from exc
        return entry.macro_action

    def is_evaluable(self, native_label: str) -> bool:
        try:
            return self.entries[native_label].include_in_evaluation
        except KeyError as exc:
            raise ActionMappingError(f"Unknown native action label: {native_label!r}") from exc

    def validate_inventory(self, native_labels: Iterable[str]) -> dict[str, int]:
        counts = Counter(native_labels)
        unknown = sorted(set(counts) - set(self.entries))
        unused = sorted(set(self.entries) - set(counts))
        if unknown:
            raise ActionMappingError(f"Unmapped native labels: {unknown}")
        if unused:
            raise ActionMappingError(f"Mapping contains labels absent from data: {unused}")
        return dict(sorted(counts.items()))
