#!/usr/bin/env python3
"""Write a Nexar candidate list for manual annotation.

The output never contains an assigned action. Metadata must already be on disk;
this script does not download the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskvla.data.annotation_sample import propose_annotation_candidates
from riskvla.data.nexar import NexarDataset, NexarRecord


def _load(path: Path | None, *, source_split: str, collision_label: int) -> list[NexarRecord]:
    if path is None:
        return []
    if not path.is_file():
        raise SystemExit(
            "Nexar metadata is not on disk (NOT_DOWNLOADED). "
            "Refusing to invent a candidate list."
        )
    return list(
        NexarDataset.from_metadata_csv(
            path,
            source_split=source_split,
            collision_label=collision_label,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--positive-csv", type=Path)
    parser.add_argument("--negative-csv", type=Path)
    parser.add_argument("--target-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.positive_csv is None and args.negative_csv is None:
        raise SystemExit("Provide --positive-csv and/or --negative-csv. No download is performed.")
    records = _load(args.positive_csv, source_split=args.source_split, collision_label=1)
    records.extend(_load(args.negative_csv, source_split=args.source_split, collision_label=0))
    proposal = propose_annotation_candidates(
        records,
        target_size=args.target_size,
        seed=args.seed,
        source_split=args.source_split,
    )
    if proposal["actions_assigned"] or any(
        item["action"] is not None for item in proposal["candidates"]
    ):
        raise SystemExit("Refusing to write a proposal that assigns actions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {proposal['selected_records']} unlabeled candidates to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
