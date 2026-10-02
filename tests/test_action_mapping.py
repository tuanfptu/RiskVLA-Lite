from __future__ import annotations

import json
from pathlib import Path

import pytest

from riskvla.constants import ACTIONS
from riskvla.data.drama_x import DramaXDataset
from riskvla.data.labels import ActionMapping, ActionMappingError

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_mapping_covers_verified_inventory() -> None:
    mapping = ActionMapping.from_yaml(ROOT / "configs/action_mapping.yaml")
    mapping.require_frozen()
    inventory = json.loads(
        (ROOT / "artifacts/label_inventory.json").read_text(encoding="utf-8")
    )
    expanded = [
        native_label
        for native_label, count in inventory.items()
        for _ in range(count)
    ]
    assert mapping.validate_inventory(expanded) == dict(sorted(inventory.items()))
    assert set(mapping.severity) == set(ACTIONS)
    assert sum(inventory.values()) == 5_686


def test_missing_label_is_not_coerced_to_maintain() -> None:
    mapping = ActionMapping.from_yaml(ROOT / "configs/action_mapping.yaml")
    assert mapping.map_label("N/A") is None
    assert not mapping.is_evaluable("N/A")


def test_fixture_preserves_native_and_maps_every_observed_label() -> None:
    mapping = ActionMapping.from_yaml(ROOT / "configs/action_mapping.yaml")
    dataset = DramaXDataset.from_jsonl(
        ROOT / "tests/fixtures/drama_x_synthetic.jsonl",
        action_mapping=mapping,
    )
    assert len(dataset) == 9
    assert {record.native_action for record in dataset} == set(mapping.entries)
    assert sum(record.include_in_evaluation for record in dataset) == 8


def test_unknown_native_label_fails_closed() -> None:
    mapping = ActionMapping.from_yaml(ROOT / "configs/action_mapping.yaml")
    with pytest.raises(ActionMappingError, match="Unknown native"):
        mapping.map_label("Drive faster")
