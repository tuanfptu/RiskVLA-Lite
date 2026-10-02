# LEGACY / ABANDONED FOR CURRENT STUDY

DRAMA-X is not the primary dataset. This note is retained as a historical
access audit. It does not contain a private download URL. The active dataset
report is `docs/NEXAR_DATA_REPORT.md`.

# DRAMA-X Data Access Report

Checked: 2026-10-02

## VERIFIED

The [official DRAMA-X repository](https://github.com/taco-group/DRAMA-X)
points to the canonical Hugging Face dataset
[`mgod96/DRAMA-X`](https://huggingface.co/datasets/mgod96/DRAMA-X).
The pinned annotation artifact is:

```text
file: drama_x_annotated.jsonl
revision: 34c3bf70d39b2cb003f4269bfc43aa2610e6073e
sha256: e9b50168f6a34aadeef2169f82fc264d44725db0f0e3591771ad3e2e0a8f95da
valid rows: 5,686
unique ids: 5,686
```

The public dataset server exposes one `train` partition only. That name is a
Hugging Face storage split, not an authored research train split.

Every public row contains these top-level fields:

```text
image_path
video_path
Risk
Pedestrians
Cyclists
suggested_action
id
```

All 5,686 public `image_path` values and all 5,686 public `video_path` values
are empty strings. The public repository therefore provides annotations, not
media bytes or usable media URLs.

The official population script copies `s3_fileUrl` and
`s3_instructionReference` from `integrated_output_v2.json` into those fields.
It does not download the assets. The required metadata/media package is
provided through the
[Honda Research Institute DRAMA page](https://usa.honda-ri.com/drama) after an
approved [dataset request](https://usa.honda-ri.com/dataset-request-form?dataset=drama).
The request requires an eligible research affiliation and agreement to Honda's
data terms.

## Native action inventory

The checked-in inventory was calculated over every canonical row:

| Native value | Count |
|---|---:|
| `(must) Stop` | 1,533 |
| `be aware or cautious (of the important object, in case it might effect in future but no direct influence)` | 1,131 |
| `Carefully manoeuvre (around the important object)` | 1,033 |
| `Slow down` | 1,010 |
| `Follow the vehicle ahead` | 471 |
| `Yield` | 262 |
| `N/A` | 157 |
| `Start moving` | 82 |
| `Accelerate` | 7 |
| **Total** | **5,686** |

The historical mapping now lives at
`configs/legacy/drama_x_action_mapping.yaml`. It was frozen before any
DRAMA-X model evaluation. `N/A` was excluded because it is missing target
information, not a synonym for `MAINTAIN`. It is not Nexar action ground truth.

## Split and timing metadata

The public JSONL has no documented:

- source recording/session identifier;
- authoritative clip group;
- train/validation/test assignment;
- event timestamp;
- prediction timestamp;
- persistent actor track.

Some IDs share a `clip_*` textual prefix, but the source does not document that
prefix as a recording group. Numeric filename components are not treated as
timestamps. RiskVLA-Lite therefore refuses to create benchmark splits from the
public file alone.

DRAMA-X is described as a single-frame benchmark even though video was used
upstream during annotation generation. It can support action-selection quality
once images are accessible; it cannot, from the public metadata alone, support
an honest action-lead-time result.

## Licenses and handling

- DRAMA-X repository code: MIT.
- Hugging Face annotation card: CC BY 4.0.
- Honda DRAMA media: Honda's Data Sharing Agreement, including non-commercial
  research and redistribution restrictions.

These scopes are not interchangeable. Media and populated private metadata
must remain outside Git. Publication/redistribution questions should be
confirmed with the dataset owner.

## Supplied download URL

`DRAMA_DOWNLOAD_URL` is present and was not written into Git. With a browser
user agent, the URL returns HTTP 200 and the Honda Research Institute page
titled "Dataset Download Page". The default Python user agent receives HTTP
403 from that HTML page.

The page's dataset-sharing request API returned HTTP 200:

- dataset name: `DRAMA Dataset`
- request expiry: `2026-10-08T00:00:00Z`
- one listed object: `drama.tar.gz`

A one-byte ranged GET of that object, after the portal redirect, returned
HTTP 206 from `s3.us-west-2.amazonaws.com` with
`content-type: application/x-tar` and
`content-range: bytes 0-0/979616585993`. A HEAD to the same redirected object
returned HTTP 403. The archive is 979,616,585,993 bytes (912.34 GiB). One
byte was read. The archive was not downloaded and is not present on this
machine.

## BLOCKED

The approved `drama.tar.gz` object responds, but benchmark media loading,
media deduplication, source-group verification, and temporal evaluation remain
blocked. The archive was not retrieved, so `integrated_output_v2.json` and
licensed media are not on disk. All 5,686 public `image_path` and `video_path`
values remain empty. No authoritative source group has been derived. No
unrelated media were substituted.

The Honda request expires at `2026-10-08T00:00:00Z`.

## Access completion checklist

1. The supplied Honda download URL is live and the `drama.tar.gz` object
   responds. Retrieve it outside this repository before
   `2026-10-08T00:00:00Z`.
2. Store the package outside this repository.
3. Run the canonical population script against
   `drama_x_annotated.jsonl` (the model card's `.json` example is stale).
4. Inspect URL/path liveness and hashes without redistributing assets.
5. Add an owner-documented source-group mapping or official split.
6. If this legacy dataset is ever revisited, keep any DRAMA split note separate. Do not replace the active Nexar `docs/DATA_SPLIT_REPORT.md`.
7. Keep event-time evaluation disabled unless verified timestamps are added.
