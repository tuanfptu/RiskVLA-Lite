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

The mapping in `configs/action_mapping.yaml` is semantic, deterministic, and
frozen before model evaluation. `N/A` is excluded because it is missing target
information, not a synonym for `MAINTAIN`.

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

## Private download recheck: UNVERIFIED

On 2026-10-02 the running Cloud Agent process was checked for
`DRAMA_DOWNLOAD_URL` without printing or logging its value. The variable was
absent (`set=false`). No request was sent, so HTTP status, redirects, content
type, content length, filename, and archive structure were not observed.

This is not evidence that Honda access was denied. Runtime Secrets are injected
when an agent starts; this already-running agent did not receive the variable.
The full dataset was intentionally not downloaded.

| Check | Status |
|---|---|
| Honda download URL reachability | **UNVERIFIED** |
| Archive structure | **NOT INSPECTED** |
| Media bytes | **NOT DOWNLOADED** |
| Source grouping and benchmark splits | **BLOCKED** |

## BLOCKED

Benchmark media loading, media deduplication, source-group verification, and
temporal evaluation remain blocked. They require the private package to be
reachable from a process that actually has the download credential, followed by
a selective, non-committed inspection.

## Access completion checklist

1. Obtain approval through the official Honda request flow.
2. Store the package outside this repository.
3. Run the canonical population script against
   `drama_x_annotated.jsonl` (the model card's `.json` example is stale).
4. Inspect URL/path liveness and hashes without redistributing assets.
5. Add an owner-documented source-group mapping or official split.
6. Regenerate split manifests and `docs/DATA_SPLIT_REPORT.md`.
7. Keep event-time evaluation disabled unless verified timestamps are added.
