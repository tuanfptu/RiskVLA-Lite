"""Propose Nexar clips for manual action annotation.

The proposal never assigns an action. Early-hazard and false-alarm enrichment
are recorded as manual follow-ups because those judgments are not present in
the official Nexar labels.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from riskvla.data.nexar import NexarRecord

RECOMMENDED_MIN = 300
RECOMMENDED_MAX = 500


class AnnotationSampleError(ValueError):
    """Raised when a candidate sample cannot be proposed."""


def propose_annotation_candidates(
    records: Sequence[NexarRecord],
    *,
    target_size: int,
    seed: int = 42,
    source_split: str = "train",
) -> dict[str, Any]:
    """Stratify a candidate list by collision label and scene.

    ``action`` is always null. Collision labels are sampling strata, not
    driving-action ground truth.
    """
    if isinstance(target_size, bool) or not isinstance(target_size, int) or target_size <= 0:
        raise AnnotationSampleError("target_size must be a positive integer")
    pool = [record for record in records if record.source_split == source_split]
    if not pool:
        raise AnnotationSampleError(f"No Nexar records are available in split {source_split!r}")

    grouped: dict[tuple[int, str], list[NexarRecord]] = defaultdict(list)
    for record in pool:
        scene = (record.scene or "unknown").strip() or "unknown"
        grouped[(record.collision_label, scene)].append(record)

    rng = random.Random(seed)
    for items in grouped.values():
        rng.shuffle(items)
    keys = sorted(grouped)

    selected: list[NexarRecord] = []
    while len(selected) < target_size and any(grouped[key] for key in keys):
        for key in keys:
            if grouped[key] and len(selected) < target_size:
                selected.append(grouped[key].pop())

    selected.sort(key=lambda record: record.video_id)
    candidates = [
        {
            "video_id": record.video_id,
            "source_split": record.source_split,
            "collision_label": record.collision_label,
            "scene": record.scene,
            "weather": record.weather,
            "light_conditions": record.light_conditions,
            "time_of_event": record.time_of_event,
            "time_of_alert": record.time_of_alert,
            "stratum": (
                f"collision_label={record.collision_label}|"
                f"scene={record.scene or 'unknown'}"
            ),
            "annotation_status": "NOT_CREATED",
            "action": None,
        }
        for record in selected
    ]
    if any(item["action"] is not None for item in candidates):
        raise AnnotationSampleError("Candidate proposal assigned an action")

    return {
        "status": "CANDIDATES_ONLY",
        "actions_assigned": False,
        "annotation_status": "NOT_CREATED",
        "source_split": source_split,
        "seed": seed,
        "target_size": target_size,
        "recommended_range": [RECOMMENDED_MIN, RECOMMENDED_MAX],
        "outside_recommended_range": not RECOMMENDED_MIN <= target_size <= RECOMMENDED_MAX,
        "available_records": len(pool),
        "selected_records": len(candidates),
        "shortfall": max(target_size - len(candidates), 0),
        "strata": {
            "collision_label": "balanced by round-robin with scene",
            "scene": "included when official metadata provides it",
            "weather_and_light": "copied onto each candidate for later manual review",
        },
        "manual_enrichment_not_applied": [
            "early_hazard_situations",
            "false_alarm_prone_normal_scenes",
        ],
        "candidates": candidates,
    }
