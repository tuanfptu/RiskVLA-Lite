#!/usr/bin/env python3
"""Inspect pinned DRAMA-X annotations and create only group-safe splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from riskvla.data.drama_x import DramaXDataset
from riskvla.data.labels import ActionMapping
from riskvla.data.splits import create_group_splits, write_split_result

CANONICAL_REVISION = "34c3bf70d39b2cb003f4269bfc43aa2610e6073e"
CANONICAL_SHA256 = "e9b50168f6a34aadeef2169f82fc264d44725db0f0e3591771ad3e2e0a8f95da"
CANONICAL_URL = (
    "https://huggingface.co/datasets/mgod96/DRAMA-X/resolve/"
    f"{CANONICAL_REVISION}/drama_x_annotated.jsonl"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_canonical(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == CANONICAL_SHA256:
        return
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(
        CANONICAL_URL,
        headers={"User-Agent": "RiskVLA-Lite/0.1 dataset verification"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        temporary.write_bytes(response.read())
    actual = sha256(temporary)
    if actual != CANONICAL_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Canonical annotation checksum mismatch: {actual} != {CANONICAL_SHA256}"
        )
    temporary.replace(destination)


def load_inventory(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) or not isinstance(value, int)
        for key, value in payload.items()
    ):
        raise ValueError(f"Invalid label inventory: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/drama_x_annotated.jsonl"),
    )
    parser.add_argument("--download-canonical", action="store_true")
    parser.add_argument(
        "--mapping", type=Path, default=Path("configs/action_mapping.yaml")
    )
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--check-inventory", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--create-splits", type=Path)
    parser.add_argument("--group-field")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.download_canonical:
        download_canonical(args.annotations)
    mapping = ActionMapping.from_yaml(args.mapping)
    mapping.require_frozen()
    dataset = DramaXDataset.from_jsonl(args.annotations, action_mapping=mapping)
    inventory = dataset.inventory()
    mapping.validate_inventory(record.native_action for record in dataset)

    if args.inventory:
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
        )
    if args.check_inventory:
        expected = load_inventory(args.check_inventory)
        if inventory != expected:
            missing = set(expected) - set(inventory)
            added = set(inventory) - set(expected)
            changed = {
                key: (expected[key], inventory[key])
                for key in set(expected) & set(inventory)
                if expected[key] != inventory[key]
            }
            raise RuntimeError(
                "Label inventory differs from frozen evidence: "
                f"missing={sorted(missing)}, added={sorted(added)}, changed={changed}"
            )

    summary: dict[str, Any] = dataset.summary(group_field=args.group_field)
    summary["status"] = "VERIFIED"
    summary["mapping_version"] = mapping.version
    summary["macro_actions"] = dict(
        sorted(
            Counter(
                record.macro_action
                for record in dataset
                if record.include_in_evaluation
            ).items()
        )
    )
    summary["excluded_native_actions"] = dict(
        sorted(
            Counter(
                record.native_action
                for record in dataset
                if not record.include_in_evaluation
            ).items()
        )
    )

    if args.create_splits:
        split_version = (
            f"dramax-{dataset.sha256[:8]}-mapping-{mapping.version}-seed-{args.seed}"
        )
        result = create_group_splits(
            dataset,
            seed=args.seed,
            group_field=args.group_field,
            split_version=split_version,
            source_sha256=dataset.sha256,
        )
        write_split_result(result, args.create_splits)
        summary["split_report"] = result.report

    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
