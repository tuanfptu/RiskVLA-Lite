# Data Split Report

## Status: BLOCKED

No benchmark train/validation/test split is claimed.

The canonical public DRAMA-X file contains 5,686 unique sample IDs but no
authoritative source-video, recording-session, sequence-group, event-time, or
official split field. Both media path columns are empty. A random row split
could place correlated frames or derived examples from the same source in
different partitions.

The committed manifests under `data/splits/` intentionally contain:

```json
{
  "status": "BLOCKED",
  "samples": []
}
```

They are machine-readable blockers, not empty experimental results.

## Frozen split policy

- Seed: 42.
- Target fractions once groups exist: train 70%, validation 15%, test 15%.
- Unit of assignment: verified source video/clip/sequence group.
- All rows in one group must stay in one partition.
- Group names must come from an explicit field or populated source-video
  reference. Undocumented parsing of unique image IDs is disabled.
- The generated report must include sample, group, native-label, macro-action,
  and risk-label distributions.
- Mapping, split code, seed, source checksum, and split version are frozen
  before final test evaluation.

## Required evidence to unblock

One of the following is needed:

1. an official DRAMA-X train/validation/test manifest;
2. an owner-documented mapping from row ID to source recording; or
3. populated media metadata with a stable source-video/clip reference whose
   grouping semantics are confirmed.

After access, run:

```bash
python scripts/inspect_dataset.py \
  --annotations data/raw/drama_x_populated.jsonl \
  --mapping configs/action_mapping.yaml \
  --create-splits data/splits
```

The split builder validates zero group overlap and is deterministic. It fails
closed if any eligible row lacks a defensible group.

## Temporal limitation

DRAMA-X's public rows have no verified event timestamp. These splits, once
unblocked, support action-selection metrics but do not by themselves support
action lead time. A separate temporal dataset or owner-supplied timing metadata
is required for that endpoint.
