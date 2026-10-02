"""Deterministic, group-safe dataset splitting."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SplitError(ValueError):
    """Raised when a leakage-safe split cannot be created."""


SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_SEED = 42


@dataclass(frozen=True)
class GroupedSample:
    """One evaluation row whose split is inherited from ``group_id``.

    Every cutoff derived from one source video must share that video's group.
    """

    sample_id: str
    group_id: str
    label: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.sample_id).strip():
            raise SplitError("sample_id must be non-empty")
        if not str(self.group_id).strip():
            raise SplitError(
                f"sample {self.sample_id!r} lacks a defensible group"
            )
        if not str(self.label).strip():
            raise SplitError(f"sample {self.sample_id!r} lacks a label")
        object.__setattr__(self, "attributes", dict(self.attributes))

    def to_manifest_item(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": self.sample_id,
            "group_id": self.group_id,
            "label": self.label,
        }
        for key, value in self.attributes.items():
            if key in item:
                raise SplitError(f"Attribute {key!r} collides with a manifest field")
            item[key] = value
        return item


@dataclass(frozen=True)
class SplitResult:
    manifests: dict[str, dict[str, Any]]
    report: dict[str, Any]


def _validate_fractions(fractions: Mapping[str, float]) -> None:
    if set(fractions) != set(SPLIT_NAMES):
        raise SplitError(f"fractions must define exactly {SPLIT_NAMES}")
    if any(not 0 < value < 1 for value in fractions.values()):
        raise SplitError("Each split fraction must be between zero and one")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise SplitError("Split fractions must sum to one")


def assert_no_group_leakage(manifests: Mapping[str, Mapping[str, Any]]) -> None:
    groups_by_split: dict[str, set[str]] = {}
    ids_by_split: dict[str, set[str]] = {}
    for split_name in SPLIT_NAMES:
        if split_name not in manifests:
            raise SplitError(f"Missing {split_name!r} manifest")
        items = manifests[split_name].get("samples", [])
        groups_by_split[split_name] = {str(item["group_id"]) for item in items}
        ids_by_split[split_name] = {str(item["id"]) for item in items}

    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                preview = sorted(overlap)[:5]
                raise SplitError(f"Group leakage between {left} and {right}: {preview}")
            id_overlap = ids_by_split[left] & ids_by_split[right]
            if id_overlap:
                preview = sorted(id_overlap)[:5]
                raise SplitError(f"Sample leakage between {left} and {right}: {preview}")


def create_group_splits(
    samples: Iterable[GroupedSample],
    *,
    seed: int = DEFAULT_SEED,
    fractions: Mapping[str, float] | None = None,
    split_version: str = "v1",
    source_sha256: str | None = None,
    group_field: str = "video_id",
) -> SplitResult:
    """Assign whole source groups to train, validation, and test.

    Assignment greedily serves the largest normalized sample/class deficits.
    Randomization controls group order only and is fully seeded. This function
    does not write a benchmark split; callers must invoke it only after manual
    annotations exist.
    """
    split_fractions = dict(fractions or DEFAULT_FRACTIONS)
    _validate_fractions(split_fractions)
    rows = list(samples)
    if not rows:
        raise SplitError("No evaluable records are available")
    if any(not isinstance(row, GroupedSample) for row in rows):
        raise SplitError("Split inputs must be GroupedSample values")
    seen_ids: set[str] = set()
    for row in rows:
        if row.sample_id in seen_ids:
            raise SplitError(f"Duplicate sample id {row.sample_id!r}")
        seen_ids.add(row.sample_id)

    grouped: dict[str, list[GroupedSample]] = defaultdict(list)
    for row in rows:
        grouped[row.group_id].append(row)
    if len(grouped) < len(SPLIT_NAMES):
        raise SplitError("At least three independent source groups are required")

    all_labels = sorted({row.label for row in rows})
    total_by_label = Counter(row.label for row in rows)
    target_samples = {
        split: split_fractions[split] * len(rows) for split in SPLIT_NAMES
    }
    target_labels = {
        split: {
            label: split_fractions[split] * total_by_label[label]
            for label in all_labels
        }
        for split in SPLIT_NAMES
    }

    rng = random.Random(seed)
    group_order = list(grouped)
    rng.shuffle(group_order)
    group_order.sort(key=lambda group: len(grouped[group]), reverse=True)

    assigned_groups: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    assigned_samples: Counter[str] = Counter()
    assigned_labels: dict[str, Counter[str]] = {
        name: Counter() for name in SPLIT_NAMES
    }

    for group in group_order:
        group_rows = grouped[group]
        group_labels = Counter(row.label for row in group_rows)

        def deficit(split: str) -> float:
            sample_target = max(target_samples[split], 1.0)
            sample_deficit = (target_samples[split] - assigned_samples[split]) / sample_target
            label_deficits = []
            for label in all_labels:
                label_target = max(target_labels[split][label], 1.0)
                label_deficits.append(
                    (target_labels[split][label] - assigned_labels[split][label])
                    / label_target
                )
            return sample_deficit + sum(label_deficits) / max(len(label_deficits), 1)

        chosen = max(SPLIT_NAMES, key=deficit)
        assigned_groups[chosen].append(group)
        assigned_samples[chosen] += len(group_rows)
        assigned_labels[chosen].update(group_labels)

    if any(not assigned_groups[name] for name in SPLIT_NAMES):
        raise SplitError("Grouping produced an empty partition; more groups are required")

    manifests: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        items = [
            row.to_manifest_item()
            for group in assigned_groups[split_name]
            for row in grouped[group]
        ]
        items.sort(key=lambda item: item["id"])
        manifests[split_name] = {
            "status": "READY",
            "split": split_name,
            "split_version": split_version,
            "seed": seed,
            "source_sha256": source_sha256,
            "group_field": group_field,
            "samples": items,
        }

    assert_no_group_leakage(manifests)
    report_splits: dict[str, Any] = {}
    optional_keys = ("native_action", "macro_action", "risk_label", "collision_label")
    for split_name, manifest in manifests.items():
        items = manifest["samples"]
        entry: dict[str, Any] = {
            "samples": len(items),
            "groups": len({item["group_id"] for item in items}),
            "labels": dict(sorted(Counter(item["label"] for item in items).items())),
        }
        for key in optional_keys:
            if items and all(key in item for item in items):
                entry[key] = dict(sorted(Counter(item[key] for item in items).items()))
        report_splits[split_name] = entry
    return SplitResult(
        manifests=manifests,
        report={
            "status": "READY",
            "split_version": split_version,
            "seed": seed,
            "group_field": group_field,
            "source_sha256": source_sha256,
            "fractions": split_fractions,
            "splits": report_splits,
        },
    )


def write_split_result(result: SplitResult, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for split_name, manifest in result.manifests.items():
        path = destination / f"{split_name}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (destination / "report.json").write_text(
        json.dumps(result.report, indent=2) + "\n", encoding="utf-8"
    )
