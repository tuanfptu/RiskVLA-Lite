from __future__ import annotations

import json
from pathlib import Path

import pytest

from riskvla.data.drama_x import DramaXDataset, DramaXError
from riskvla.data.labels import ActionMapping

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/drama_x_synthetic.jsonl"


def test_loader_preserves_native_labels_and_reports_media() -> None:
    mapping = ActionMapping.from_yaml(ROOT / "configs/action_mapping.yaml")
    dataset = DramaXDataset.from_jsonl(FIXTURE, action_mapping=mapping)
    summary = dataset.summary()
    assert summary["records"] == 9
    assert summary["unique_ids"] == 9
    assert summary["nonempty_image_paths"] == 9
    assert summary["nonempty_video_paths"] == 9
    assert dataset[0].native_action == "(must) Stop"
    assert dataset[0].macro_action == "BRAKE_OR_STOP"
    assert any(record.native_action == "N/A" and record.macro_action is None for record in dataset)


def test_loader_rejects_missing_fields_and_duplicate_ids(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text('{"id":"only-id"}\n', encoding="utf-8")
    with pytest.raises(DramaXError, match="Missing DRAMA-X fields"):
        DramaXDataset.from_jsonl(incomplete)

    duplicated = tmp_path / "duplicate.jsonl"
    line = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    duplicated.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(DramaXError, match="Duplicate id"):
        DramaXDataset.from_jsonl(duplicated)


def test_committed_splits_are_explicitly_blocked() -> None:
    for split_name in ("train", "val", "test"):
        payload = json.loads(
            (ROOT / "data/splits" / f"{split_name}.json").read_text(encoding="utf-8")
        )
        assert payload["status"] == "BLOCKED"
        assert payload["samples"] == []
        assert payload["seed"] == 42
