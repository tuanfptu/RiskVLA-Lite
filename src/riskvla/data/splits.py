"""Deterministic, group-safe dataset splitting."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from riskvla.data.drama_x import DramaXRecord


class SplitError(ValueError):
    """Raised when a leakage-safe split cannot be created."""


SPLIT_NAMES = ("train", "val", "test")


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
    records: Iterable[DramaXRecord],
    *,
    seed: int = 42,
    fractions: Mapping[str, float] | None = None,
    group_field: str | None = None,
    split_version: str = "v1",
    source_sha256: str | None = None,
) -> SplitResult:
    """Split records while keeping every verified source group together.

    Assignment greedily serves the largest normalized sample/class deficits.
    Randomization controls group order only and is fully seeded.
    """
    split_fractions = dict(
        fractions or {"train": 0.70, "val": 0.15, "test": 0.15}
    )
    _validate_fractions(split_fractions)
    rows = [record for record in records if record.include_in_evaluation]
    if not rows:
        raise SplitError("No evaluable records are available")
    if any(record.macro_action is None for record in rows):
        raise SplitError("Every evaluable record must have a mapped macro action")

    grouped: dict[str, list[DramaXRecord]] = defaultdict(list)
    missing_group: list[str] = []
    for record in rows:
        group = record.source_group(explicit_field=group_field)
        if group is None:
            missing_group.append(record.sample_id)
        else:
            grouped[group].append(record)
    if missing_group:
        preview = ", ".join(missing_group[:5])
        raise SplitError(
            "Cannot create leakage-safe splits: "
            f"{len(missing_group)} eligible records lack a defensible group "
            f"(examples: {preview})"
        )
    if len(grouped) < len(SPLIT_NAMES):
        raise SplitError("At least three independent source groups are required")

    all_labels = sorted({record.macro_action for record in rows if record.macro_action})
    total_by_label = Counter(record.macro_action for record in rows)
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
    # Large groups first; stable shuffle resolves equal-size ties.
    group_order.sort(key=lambda group: len(grouped[group]), reverse=True)

    assigned_groups: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    assigned_samples = Counter()
    assigned_labels: dict[str, Counter[str]] = {
        name: Counter() for name in SPLIT_NAMES
    }

    for group in group_order:
        group_rows = grouped[group]
        group_labels = Counter(record.macro_action for record in group_rows)

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
            # Favor under-filled splits; deterministic name order breaks ties.
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
            record.to_manifest_item(group)
            for group in assigned_groups[split_name]
            for record in grouped[group]
        ]
        items.sort(key=lambda item: item["id"])
        manifests[split_name] = {
            "status": "READY",
            "split": split_name,
            "split_version": split_version,
            "seed": seed,
            "source_sha256": source_sha256,
            "group_field": group_field or "video_path",
            "samples": items,
        }

    assert_no_group_leakage(manifests)
    report_splits: dict[str, Any] = {}
    for split_name, manifest in manifests.items():
        items = manifest["samples"]
        report_splits[split_name] = {
            "samples": len(items),
            "groups": len({item["group_id"] for item in items}),
            "macro_actions": dict(
                sorted(Counter(item["macro_action"] for item in items).items())
            ),
            "native_actions": dict(
                sorted(Counter(item["native_action"] for item in items).items())
            ),
            "risk_labels": dict(
                sorted(Counter(item["risk_label"] for item in items).items())
            ),
        }
    return SplitResult(
        manifests=manifests,
        report={
            "status": "READY",
            "split_version": split_version,
            "seed": seed,
            "group_field": group_field or "video_path",
            "source_sha256": source_sha256,
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
