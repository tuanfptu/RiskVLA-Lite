from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from riskvla.data.drama_x import DramaXDataset
from riskvla.data.labels import ActionMapping
from riskvla.data.splits import (
    SplitError,
    assert_no_group_leakage,
    create_group_splits,
)

ROOT = Path(__file__).resolve().parents[1]


def load_fixture() -> DramaXDataset:
    mapping = ActionMapping.from_yaml(ROOT / "configs/action_mapping.yaml")
    return DramaXDataset.from_jsonl(
        ROOT / "tests/fixtures/drama_x_synthetic.jsonl",
        action_mapping=mapping,
    )


def test_group_split_is_deterministic_and_leakage_free() -> None:
    dataset = load_fixture()
    first = create_group_splits(dataset, seed=42, split_version="fixture-v1")
    second = create_group_splits(dataset, seed=42, split_version="fixture-v1")
    assert first.manifests == second.manifests
    assert_no_group_leakage(first.manifests)

    located: dict[str, str] = {}
    for split, manifest in first.manifests.items():
        for sample in manifest["samples"]:
            previous = located.setdefault(sample["group_id"], split)
            assert previous == split
    assert sum(
        len(manifest["samples"]) for manifest in first.manifests.values()
    ) == 8


def test_split_refuses_records_without_defensible_group() -> None:
    records = [
        replace(record, video_path="")
        for record in load_fixture()
        if record.include_in_evaluation
    ]
    with pytest.raises(SplitError, match="lack a defensible group"):
        create_group_splits(records)


def test_explicit_owner_group_field_is_supported() -> None:
    records = []
    for index, record in enumerate(load_fixture()):
        if not record.include_in_evaluation:
            continue
        records.append(
            replace(
                record,
                video_path="",
                extras={**record.extras, "source_sequence": f"sequence-{index // 2}"},
            )
        )
    result = create_group_splits(records, group_field="source_sequence")
    assert result.report["group_field"] == "source_sequence"
    assert_no_group_leakage(result.manifests)
