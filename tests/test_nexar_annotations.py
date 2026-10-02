from __future__ import annotations

import json
from pathlib import Path

import pytest

from riskvla.constants import ACTIONS
from riskvla.data.annotations import (
    ANNOTATION_FIELDS,
    AnnotationError,
    attach_preserved_metadata,
    validate_annotation,
)
from riskvla.data.nexar import NexarRecord
from riskvla.data.splits import (
    GroupedSample,
    SplitError,
    assert_no_group_leakage,
    create_group_splits,
)

ROOT = Path(__file__).resolve().parents[1]


def _annotation(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "video_id": "00010",
        "source_split": "train",
        "collision_label": 1,
        "time_of_event": 20.0,
        "time_of_alert": 18.5,
        "observation_cutoff": 18.0,
        "action": "SLOW",
        "actionable_from": 17.5,
        "ego_relevance": "direct",
        "confidence": 4,
        "ambiguous": False,
        "annotator": "researcher",
        "notes": "visible closing gap",
    }
    payload.update(overrides)
    return payload


def test_schema_file_matches_the_manual_action_contract() -> None:
    schema = json.loads(
        (ROOT / "data/annotations/nexar_action_schema.json").read_text(encoding="utf-8")
    )
    assert schema["required"] == list(ANNOTATION_FIELDS)
    assert schema["properties"]["action"]["enum"] == list(ACTIONS)
    assert "MANEUVER" not in schema["properties"]["action"]["enum"]
    record = validate_annotation(_annotation())
    assert record.action == "SLOW"
    assert record.sample_id == "00010@18.000"


def test_annotation_rejects_derived_or_post_event_labels() -> None:
    with pytest.raises(AnnotationError, match="Unknown action"):
        validate_annotation(_annotation(action="MANEUVER"))
    with pytest.raises(AnnotationError, match="strictly before"):
        validate_annotation(_annotation(observation_cutoff=20.0))
    with pytest.raises(AnnotationError, match="must not invent"):
        validate_annotation(
            _annotation(
                video_id="00011",
                collision_label=0,
                time_of_event=10.0,
                time_of_alert=None,
                observation_cutoff=8.0,
                action="MAINTAIN",
                actionable_from=None,
            )
        )


def test_preserved_metadata_cannot_replace_the_human_action() -> None:
    annotation = validate_annotation(
        _annotation(
            nexar_metadata={"action": "BRAKE_OR_STOP", "collision_label": 1}
        )
    )
    assert annotation.action == "SLOW"
    record = NexarRecord.from_dict(
        {
            "video_id": "00010",
            "source_split": "train",
            "label": 1,
            "video_path": "train/positive/00010.mp4",
            "time_of_event": 20.0,
            "scene": "Urban",
        }
    )
    attached = attach_preserved_metadata(annotation, record)
    assert attached.action == "SLOW"
    assert attached.nexar_metadata is not None
    assert attached.nexar_metadata["scene"] == "Urban"
    assert "action" not in attached.nexar_metadata


def test_cutoffs_from_one_video_stay_in_one_split() -> None:
    samples = []
    for video_index in range(6):
        for cutoff in (17.0, 18.0, 19.0):
            annotation = validate_annotation(
                _annotation(
                    video_id=f"{video_index:05d}",
                    observation_cutoff=cutoff,
                    actionable_from=None,
                    action="CAUTION" if video_index % 2 == 0 else "SLOW",
                )
            )
            samples.append(annotation.to_grouped_sample())
    result = create_group_splits(samples, seed=42, split_version="fixture-v1")
    assert_no_group_leakage(result.manifests)
    located: dict[str, str] = {}
    for split_name, manifest in result.manifests.items():
        groups_in_split = {item["group_id"] for item in manifest["samples"]}
        for group_id in groups_in_split:
            assert group_id not in located
            located[group_id] = split_name
        video_ids = {item["video_id"] for item in manifest["samples"]}
        assert groups_in_split == video_ids
    assert len(located) == 6
    repeated = create_group_splits(samples, seed=42, split_version="fixture-v1")
    assert result.manifests == repeated.manifests


def test_split_refuses_a_blank_group() -> None:
    with pytest.raises(SplitError, match="defensible group"):
        GroupedSample(sample_id="00010@18.000", group_id="  ", label="SLOW")
