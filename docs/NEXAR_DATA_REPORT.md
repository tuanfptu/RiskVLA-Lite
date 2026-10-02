# Nexar Data Report

Checked: 2026-10-02

Primary source:

- Dataset card and repository: <https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction>
- Hugging Face API identity: `nexar-ai/nexar_collision_prediction`
- Paper cited by the card: Daniel C. Moura, Shizhan Zhu, and Orly Zvitia, *Nexar Dashcam Collision Prediction Dataset and Challenge*, arXiv:2503.03848, 2025

## Pinned revision

| Field | Verified value |
|---|---|
| Git revision (`sha`) | `7535d0656dac31d7da2846913bb69c6331d2a70a` |
| `createdAt` | 2025-01-28T15:42:33.000Z |
| `lastModified` | 2026-09-29T14:37:45.000Z |
| Gate | `auto` (access request required) |
| License name on the card | `nexar-open-data-license` |
| `usedStorage` | 31,379,211,365 bytes (31.379 GB) |
| Card label | Total file size 31.4 GB |

`usedStorage` is the repository storage figure returned by the Hugging Face
dataset API. It matches the card's 31.4 GB label at one-decimal precision.
The public file listing does not include per-file byte sizes for this gated
repository, so split-level byte totals are not claimed.

## Verified file inventory

From the public repository listing at the pinned revision:

| Location | Videos | Notes |
|---|---:|---|
| `train/positive` | 750 | plus `metadata.csv` |
| `train/negative` | 750 | plus `metadata.csv` |
| `test-public/positive` | 334 | plus `metadata.csv` |
| `test-public/negative` | 333 | plus `metadata.csv` |
| `test-private/positive` | 338 | plus `metadata.csv` |
| `test-private/negative` | 339 | plus `metadata.csv` |
| Total videos | 2,844 | 1,500 train and 1,344 test |

Video stems are unique across the listing (2,844 distinct ids). Root files
also include `README.md`, `LICENSE`, `evaluate_submission.py`,
`sample_submission.csv`, `solution.csv`, and `time_to_accident_test_map.csv`.

This confirms the card's statement that the training folder contains 1,500
videos and that half are positive and half are negative. The test listing
has 1,344 videos, matching the paper's test-set size. The card says the test
set is divided into public and private subsets. The current listing is 667
public and 677 private videos, so those subsets are close in size but not
equal in this revision.

## Schema verified from the official card

The card's normalized training example contains:

```text
label, time_of_event, time_of_alert, light_conditions, weather, scene, time_to_accident
```

A positive example has `label: 1`, numeric `time_of_event` and
`time_of_alert`, and `time_to_accident: None`. A negative example has
`label: 0` and null event, alert, and time-to-accident fields.

The card defines `time_to_accident` as how long before the event a test video
was clipped. It says the column is not available on the unclipped training
set. Test videos are about 10 seconds and end at 500, 1,000, or 1,500 ms
before the event.

A historical public commit on the same repository (`ff786ec`, 2025-06-09)
shows a training negative `metadata.csv` header:

```text
file_name,time_of_event,time_of_alert,light_conditions,weather,scene
```

with empty event fields. That commit is evidence of an earlier header, not a
substitute for the current gated CSV bytes.

## Stated by the card, not re-measured on media

The card states that videos are 1280x720 at 30 FPS and typically last about
40 seconds. Those properties describe the published dataset, especially the
training videos. They were not re-measured here because no media were
downloaded. The card separately says test videos are about 10 seconds, so the
40-second figure must not be applied to the clipped test set.

## Metadata CSV rows: VERIFIED

The six `metadata.csv` files and `time_to_accident_test_map.csv` were read
from the pinned revision. No `.mp4` file was downloaded. Every metadata file
has the same columns:

```text
file_name, time_of_event, time_of_alert, light_conditions, weather, scene, time_to_accident
```

| File | Rows | `time_of_event` | `time_of_alert` | `time_to_accident` |
|---|---:|---|---|---|
| `train/positive` | 750 | present, 3.032–56.800 s | present, and `<= time_of_event` on all 750 rows | column present, all empty |
| `train/negative` | 750 | all empty | all empty | all empty |
| `test-public/positive` | 334 | present | present, and `<= time_of_event` | present, 0.500–1.500 s |
| `test-public/negative` | 333 | all empty | all empty | present, 0.500–1.500 s |
| `test-private/positive` | 338 | present | present, and `<= time_of_event` | present, 0.500–1.500 s |
| `test-private/negative` | 339 | all empty | all empty | present, 0.500–1.500 s |

The training `time_to_accident` column exists but is empty, which matches the
card's statement that the value is not available for unclipped training
videos. On the test set the value is populated for negative clips as well as
positive clips. For negatives it is a clip-construction field, not an event
time, and the loader does not copy it into `time_of_event`.

One training positive row and one training negative row have an empty
`weather` cell. Observed `scene` values include Urban, Highway, Sub-urban,
Rural, Industrial, Other, and, on one training positive row, Nature.
`light_conditions` values include Normal, Bright, Dark, and Twilight.

`time_to_accident_test_map.csv` has 568 rows and columns named `0.5`, `1.0`,
and `1.5`. Those columns contain 568, 464, and 312 mutually exclusive video
ids (1,344 total), which are the official test clips. This matches the
paper's description that the test set is built from 568 source videos, with
up to three crops each. Those crops are not independent split units.

## Still not re-measured on media

- Per-file video byte sizes (the authenticated listing still omitted them)
- Actual frame rate, resolution, and duration of any clip

The loader accepts the verified CSV columns and the card's normalized fields.
It does not invent `time_to_accident` when the cell is empty.

## What this dataset does not provide

Nexar does not provide human driving actions such as `MAINTAIN`, `CAUTION`,
`SLOW`, or `BRAKE_OR_STOP`. Collision labels, `time_of_event`,
`time_of_alert`, BADAS scores, and model outputs are not action ground truth.
The action subset is specified in
[`NEXAR_ACTION_ANNOTATION_PROTOCOL.md`](NEXAR_ACTION_ANNOTATION_PROTOCOL.md).

## Access status

| Check | Status |
|---|---|
| Dataset identity, revision, and file inventory | **VERIFIED** |
| Card schema and train class balance | **VERIFIED** |
| Metadata CSV rows at the pinned revision | **VERIFIED** |
| Media | **NOT DOWNLOADED** |

Machine-readable status: [`outputs/blockers/nexar_media.json`](../outputs/blockers/nexar_media.json).
