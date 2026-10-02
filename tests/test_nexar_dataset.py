from __future__ import annotations

import json
from pathlib import Path

import pytest

from riskvla.data.annotation_sample import propose_annotation_candidates
from riskvla.data.nexar import (
    DATASET_REVISION,
    USED_STORAGE_BYTES,
    NexarDataset,
    NexarError,
    NexarRecord,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/nexar_synthetic.jsonl"


def test_jsonl_normalizes_positive_and_negative_records() -> None:
    dataset = NexarDataset.from_jsonl(FIXTURE)
    by_id = {record.video_id: record for record in dataset}
    positive = by_id["00010"]
    negative = by_id["00011"]
    assert positive.collision_label == 1
    assert positive.time_of_event == pytest.approx(20.0)
    assert positive.time_of_alert == pytest.approx(18.5)
    assert positive.scene == "Urban"
    assert positive.time_to_accident is None
    assert negative.collision_label == 0
    assert negative.time_of_event is None
    assert negative.time_of_alert is None
    assert negative.weather == "Rain"
    clipped = by_id["00200"]
    assert clipped.source_split == "test-public"
    assert clipped.time_to_accident == pytest.approx(0.5)
    preserved = positive.preserved_metadata()
    assert preserved["dataset_revision"] == DATASET_REVISION
    assert "action" not in preserved


def test_metadata_csv_does_not_require_video_files() -> None:
    positive = NexarDataset.from_metadata_csv(
        ROOT / "tests/fixtures/nexar_positive_metadata.csv",
        source_split="train",
        collision_label=1,
    )
    negative = NexarDataset.from_metadata_csv(
        ROOT / "tests/fixtures/nexar_negative_metadata.csv",
        source_split="train",
        collision_label=0,
    )
    assert positive[0].video_id == "00010"
    assert positive[0].video_path == "train/positive/00010.mp4"
    assert positive[0].time_of_event == pytest.approx(20.367)
    assert negative[0].time_of_event is None
    assert negative[0].video_path == "train/negative/00011.mp4"
    assert not Path(positive[0].video_path).is_file()


def test_negative_time_to_accident_is_not_an_event() -> None:
    record = NexarRecord.from_dict(
        {
            "file_name": "01044.mp4",
            "source_split": "test-public",
            "label": 0,
            "time_of_event": "",
            "time_of_alert": "",
            "time_to_accident": "1.0",
            "weather": "",
            "scene": "Urban",
        }
    )
    assert record.collision_label == 0
    assert record.time_of_event is None
    assert record.time_to_accident == pytest.approx(1.0)
    assert record.weather is None


def test_negative_event_time_and_missing_positive_event_are_rejected() -> None:
    with pytest.raises(NexarError, match="must not carry time_of_event"):
        NexarRecord.from_dict(
            {
                "video_id": "00011",
                "source_split": "train",
                "label": 0,
                "video_path": "train/negative/00011.mp4",
                "time_of_event": 10.0,
            }
        )
    with pytest.raises(NexarError, match="requires time_of_event"):
        NexarRecord.from_dict(
            {
                "video_id": "00010",
                "source_split": "train",
                "label": 1,
                "video_path": "train/positive/00010.mp4",
                "time_of_event": None,
            }
        )


def test_candidate_sample_does_not_assign_actions() -> None:
    dataset = NexarDataset.from_jsonl(FIXTURE)
    proposal = propose_annotation_candidates(
        [record for record in dataset if record.source_split == "train"],
        target_size=4,
        seed=42,
    )
    assert proposal["actions_assigned"] is False
    assert proposal["annotation_status"] == "NOT_CREATED"
    assert {item["collision_label"] for item in proposal["candidates"]} == {0, 1}
    assert all(item["action"] is None for item in proposal["candidates"])
    assert "false_alarm_prone_normal_scenes" in proposal["manual_enrichment_not_applied"]


def test_published_storage_figure_is_the_verified_byte_count() -> None:
    blocker = json.loads(
        (ROOT / "outputs/blockers/nexar_media.json").read_text(encoding="utf-8")
    )
    assert blocker["used_storage_bytes"] == USED_STORAGE_BYTES
    assert blocker["revision"] == DATASET_REVISION
    assert blocker["media"] == "NOT_DOWNLOADED"
    assert blocker["metadata_access"] == "VERIFIED"
    assert blocker["metrics"] is None
