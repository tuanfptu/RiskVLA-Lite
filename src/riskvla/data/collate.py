"""Small, framework-neutral batch collation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def collate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Transpose homogeneous record dictionaries without importing Torch."""
    if not records:
        raise ValueError("Cannot collate an empty batch")
    keys = set(records[0])
    for index, record in enumerate(records[1:], start=1):
        if set(record) != keys:
            raise ValueError(f"Record {index} has different keys")
    return {key: [record[key] for record in records] for key in sorted(keys)}
