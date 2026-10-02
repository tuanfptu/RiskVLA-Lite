"""Primary four-class Nexar action schema.

Human annotations already use these names. This file does not map collision
labels, BADAS scores, or DRAMA-X native labels onto actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from riskvla.constants import ACTIONS, DEFAULT_SEVERITY, EXTENDED_ACTIONS


class ActionSchemaError(ValueError):
    """Raised when the active action schema is missing or inconsistent."""


@dataclass(frozen=True)
class NexarActionSchema:
    version: str
    status: str
    frozen_before_test_evaluation: bool
    actions: tuple[str, ...]
    severity: dict[str, int]
    label_source: str
    annotations_status: str
    extended_actions: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> NexarActionSchema:
        schema_path = Path(path)
        with schema_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ActionSchemaError("Action schema root must be a mapping")
        actions = tuple(raw.get("actions", ()))
        if actions != ACTIONS:
            raise ActionSchemaError(
                f"Primary actions must be exactly {ACTIONS} in that order"
            )
        severity = raw.get("severity")
        if not isinstance(severity, dict) or set(severity) != set(ACTIONS):
            raise ActionSchemaError("severity must define every primary action")
        normalized: dict[str, int] = {}
        for action, value in severity.items():
            if not isinstance(value, int) or value < 0:
                raise ActionSchemaError(f"Invalid severity for {action!r}: {value!r}")
            normalized[str(action)] = value
        if normalized != DEFAULT_SEVERITY:
            raise ActionSchemaError("severity does not match the primary action ranks")
        extended = tuple(raw.get("extended_actions_not_in_primary_benchmark", ()))
        if extended != ("MANEUVER",) or set(ACTIONS).intersection(extended):
            raise ActionSchemaError(
                "MANEUVER must stay outside the primary benchmark and inside the extension list"
            )
        if set(EXTENDED_ACTIONS) != set(ACTIONS).union(extended):
            raise ActionSchemaError("Extended action list is inconsistent")
        label_source = raw.get("label_source")
        if label_source != "human_annotation":
            raise ActionSchemaError("Nexar actions must come from human annotation")
        annotations_status = str(raw.get("annotations_status", ""))
        return cls(
            version=str(raw.get("version", "")),
            status=str(raw.get("status", "")),
            frozen_before_test_evaluation=bool(
                raw.get("frozen_before_test_evaluation", False)
            ),
            actions=actions,
            severity=normalized,
            label_source=label_source,
            annotations_status=annotations_status,
            extended_actions=extended,
        )

    def require_frozen(self) -> None:
        if self.status != "frozen" or not self.frozen_before_test_evaluation:
            raise ActionSchemaError(
                "Action schema must be frozen before benchmark evaluation"
            )
        if not self.version:
            raise ActionSchemaError("Frozen action schema requires a version")

    def validate_action(self, action: str) -> str:
        if action not in self.actions:
            raise ActionSchemaError(f"Unknown action {action!r}; expected one of {self.actions}")
        return action

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "actions": list(self.actions),
            "severity": self.severity,
            "label_source": self.label_source,
            "annotations_status": self.annotations_status,
            "extended_actions_not_in_primary_benchmark": list(self.extended_actions),
        }
