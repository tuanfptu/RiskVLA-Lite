# RiskVLA-Lite

**Temporal Risk-Conditioned Vision-Language-Action for Early Safety Action
Selection**

> Research prototype only. Not a certified driver-assistance or
> vehicle-control system.

Primary dataset: [`nexar-ai/nexar_collision_prediction`](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction).

Research question:

> Can temporal ego-collision risk signals improve the accuracy, timeliness,
> and efficiency of a lightweight vision-language action selector on
> real-world dashcam safety scenarios?

Primary hypothesis, not yet measured:

> Temporal BADAS risk conditioning improves safety-action selection relative
> to the same Qwen3-VL-2B model using visual evidence alone.

```text
Nexar dashcam video
        |
        +------ BADAS-Open (frozen)
        |           |
        |           +--> R(t)
        |           +--> risk slope
        |           +--> recent peak
        |
        +------ pre-cutoff visual frames
                        |
                        v
                  Qwen3-VL-2B
                        +
                  temporal risk
                        |
                        v
                 SAFETY ACTION
```

Actions for the first study:

`MAINTAIN`, `CAUTION`, `SLOW`, `BRAKE_OR_STOP`.

BADAS answers when collision risk is emerging. RiskVLA answers what safety
action the ego vehicle should take. `MANEUVER` is reserved for a later
extension and is not part of this benchmark.

## Current evidence status

| Component | Status | Evidence |
|---|---|---|
| Nexar identity, revision, file inventory, and metadata rows | **VERIFIED** | Card, listing, and metadata CSVs at `7535d0656dac31d7da2846913bb69c6331d2a70a` |
| Nexar media | **NOT DOWNLOADED** | [`outputs/blockers/nexar_media.json`](outputs/blockers/nexar_media.json) |
| Human action annotations | **NOT CREATED** | Schema only; collision labels are not actions |
| Action split | **NOT CREATED** | Official Nexar train/test is not the action split |
| BADAS source contract | **VERIFIED** | Official source and checkpoint contract inspected |
| BADAS checkpoint access | **VERIFIED** | Authenticated HEAD of `weights/badas_open.pth` (3,979,436,545 bytes); body not downloaded |
| BADAS GPU runtime | **BLOCKED** | No CUDA GPU in this development environment |
| Qwen GPU runtime | **BLOCKED / NOT RUN** | Weights were not loaded here |
| A/B/C results | **NOT MEASURED** | [`docs/RESULTS.md`](docs/RESULTS.md) |
| LoRA | **NOT STARTED** | Deferred until zero-shot A/B/C exists |

DRAMA-X is legacy. Its notes live in [`docs/legacy/drama_x/`](docs/legacy/drama_x/).
It is not an active blocker and `DRAMA_DOWNLOAD_URL` is not a project secret.

## Ablations

A, B, and C see the same pre-cutoff frames.

- **A — Visual only:** four frames.
- **B — Current risk:** those frames plus `R_t`.
- **C — Temporal risk:** those frames plus `R_t`, risk slope, and recent peak.

Risk-centered frame selection is a separate experiment because it changes the
visual evidence. Eight frames are a later comparison, not the default.

The parser accepts only this object:

```json
{"action": "SLOW", "confidence": 0.84}
```

Invalid output stays invalid. It is not coerced into a nearby action.

## Install

Python 3.10–3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install `".[inference]"`, `".[train]"`, or `".[demo]"` only on the machine
that needs them. Compatibility notes are in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Workflow

### 1. Inspect metadata without downloading media

The pinned facts are in [`docs/NEXAR_DATA_REPORT.md`](docs/NEXAR_DATA_REPORT.md).
A dry run prints the published disk requirement and does not transfer files:

```bash
python scripts/download_nexar.py --output /path/to/nexar --dry-run
```

On the persistent machine, after `HF_TOKEN` is set in the environment:

```bash
python scripts/download_nexar.py --output /path/to/nexar
```

The dataset root is an argument. Library code does not assume a machine path.
Do not commit videos or the token.

### 2. Annotate a human subset

Follow [`docs/NEXAR_ACTION_ANNOTATION_PROTOCOL.md`](docs/NEXAR_ACTION_ANNOTATION_PROTOCOL.md).
The schema is [`data/annotations/nexar_action_schema.json`](data/annotations/nexar_action_schema.json).
Candidate sampling can propose clip ids. It does not write actions.

### 3. Split only after those labels exist

The policy is 70% / 15% / 15% with seed 42, grouped by source video. Do not
create the split before the annotations exist. See
[`docs/DATA_SPLIT_REPORT.md`](docs/DATA_SPLIT_REPORT.md).

### 4. Run zero-shot A/B/C before any training

BADAS stays frozen. LoRA is not started. Thresholds for selective invocation
come from validation only.

```bash
pytest -q
ruff check src tests scripts
```

## Repository guide

- [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md)
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md)
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)
- [`docs/RESULTS.md`](docs/RESULTS.md)
- [`src/riskvla`](src/riskvla)

## License and data terms

Project code is Apache-2.0. See [`LICENSE`](LICENSE). Nexar media and metadata
are covered by the Nexar Open Data License on the dataset repository. BADAS
and Qwen keep their own terms. Do not commit media, checkpoints, or tokens.
