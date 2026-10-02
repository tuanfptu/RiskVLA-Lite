# Data Split Report

## Status: NOT CREATED

No action-benchmark train, validation, or test split is claimed.

`data/splits/` contains empty `NOT_CREATED` manifests. They are blockers, not
experimental results. The suggested fractions below are policy constants in
`configs/nexar_split_policy.yaml`. They have not been applied to data.

## Why the official Nexar divisions are not the action split

The official dataset has `train`, `test-public`, and `test-private`. Those
divisions serve collision-time prediction:

- Training videos are the long, unclipped clips. The card describes them as
  typically about 40 seconds, with `time_of_event` on positive cases.
- Test videos are about 10 seconds and already end 0.5, 1.0, or 1.5 seconds
  before an event. The paper describes up to three crops from one source video.
- `time_to_accident_test_map.csv` confirms that structure at this revision:
  568 rows and columns `0.5`, `1.0`, and `1.5` hold 568, 464, and 312
  mutually exclusive test ids (1,344 clips).
- The official label is collision versus normal driving, not a safety action.
- This study's labels will be human actions on explicitly stored cutoffs.

Using the official test set as our test split would mix a different clip
construction and a different task into the action benchmark. It would also
place derivative crops of one source video on opposite sides of a split
unless the map's row is the group. The first human subset therefore stays
on official training videos, where each file is its own source clip.

The first human subset should therefore be drawn from official **train**
videos, which are one file per clip and still contain the pre-event context.
Our own split is applied only after the actions exist.

## Policy, once annotations exist

- Seed: 42.
- Fractions: train 70%, validation 15%, test 15%.
- Assignment unit: source `video_id`.
- Every cutoff derived from one video stays in that video's split.
- Thresholds are chosen on validation only.
- Official Nexar test subsets are not reused as this action split unless a
  later, verified source-video map says that is safe.

The splitter refuses a blank group and checks that no `video_id` appears in
more than one partition. It is deterministic for a fixed seed and input.
