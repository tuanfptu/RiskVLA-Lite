# RiskVLA-Lite

**Temporal Risk-Conditioned Vision-Language-Action for Early Safety Action
Selection**

> Research prototype only. Not a certified driver-assistance or
> vehicle-control system.

RiskVLA-Lite studies whether temporal output from a frozen accident-
anticipation model adds measurable value to a lightweight visual-language
model that selects a discrete safety action.

```text
Dashcam
  -> Frozen risk detector (WHEN is danger emerging?)
  -> Current and temporal risk features
  -> Selective lightweight VLA (WHAT should the ego vehicle do?)
  -> MAINTAIN | CAUTION | SLOW | BRAKE_OR_STOP | MANEUVER
```

The exact hypothesis is:

> Conditioning Qwen3-VL-2B-Instruct on temporal collision-risk signals will
> improve critical-action recall and/or action timing while reducing
> inappropriate actions relative to an otherwise identical visual-only
> baseline.

This is falsifiable. The repository does not currently claim that the
hypothesis is supported.

## Current evidence status

| Component | Status | Evidence |
|---|---|---|
| DRAMA-X annotation schema and labels | **VERIFIED** | 5,686 canonical public annotation rows and nine native labels |
| DRAMA-X private download | **UNVERIFIED** | Credential was not injected into this running agent; media are **NOT DOWNLOADED** |
| DRAMA-X grouping | **BLOCKED** | [`outputs/blockers/drama_media.json`](outputs/blockers/drama_media.json) |
| BADAS source integration | **VERIFIED** | Official source and checkpoint contract inspected |
| BADAS checkpoint access | **UNVERIFIED** | No authenticated request was sent from this running agent |
| BADAS runtime | **BLOCKED** | No CUDA GPU; [`outputs/blockers/badas_runtime.json`](outputs/blockers/badas_runtime.json) |
| Qwen integration | **INFRASTRUCTURE-READY** | Public model API implemented; weights not run on this machine |
| A/B/C research results | **NOT MEASURED** | Requires media, model runtime, and valid splits |

See [data access](docs/DATA_ACCESS_REPORT.md), [BADAS integration](docs/BADAS_INTEGRATION_REPORT.md),
and [results](docs/RESULTS.md) before interpreting project status.

## Ablations

- **A — Visual only:** four pre-cutoff frames.
- **B — Current risk:** the same frames plus `R_t`.
- **C — Temporal risk:** the same frames plus `R_t`, risk slope, and recent
  peak.

All variants share a strict JSON action parser. Invalid output is recorded, not
silently coerced. The native DRAMA-X labels remain in every processed record;
the frozen mapping is in [`configs/action_mapping.yaml`](configs/action_mapping.yaml).

## Install

Python 3.10–3.12 is supported by the project code. Create a clean environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install only the extras required for the target machine:

```bash
# Qwen/BADAS inference
python -m pip install -e ".[inference]"

# action-head or LoRA experiments
python -m pip install -e ".[train]"

# dashboard
python -m pip install -e ".[demo]"
```

The tested/target compatibility matrix and BADAS-specific caveats are in
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md). Do not install or download gated
assets by embedding credentials in commands or config files.

## Reproducible workflow

### 1. Inspect the environment and public annotations

```bash
python scripts/inspect_environment.py
python scripts/inspect_dataset.py \
  --download-canonical \
  --annotations data/raw/drama_x_annotated.jsonl \
  --inventory artifacts/label_inventory.generated.json
```

The download is pinned and checksum-verified. Compare the generated inventory
with the frozen committed inventory:

```bash
python scripts/inspect_dataset.py \
  --annotations data/raw/drama_x_annotated.jsonl \
  --mapping configs/action_mapping.yaml \
  --check-inventory artifacts/label_inventory.json
```

### 2. Create splits only after access is populated

```bash
python scripts/inspect_dataset.py \
  --annotations data/raw/drama_x_populated.jsonl \
  --mapping configs/action_mapping.yaml \
  --create-splits data/splits
```

The command fails closed if it cannot derive a defensible source group. It does
not split unique frame IDs at random.

### 3. Smoke-test BADAS

After the official Hugging Face gate is approved, set `HF_TOKEN` in the shell
and provide a licensed real video:

```bash
python scripts/run_badas_smoke.py --video /secure/path/example.mp4
```

The script records model/checkpoint revisions, shape, sampled FPS, device,
latency, peak VRAM, warm-up validity, and score bounds. No BADAS measurement is
reported until this succeeds.

### 4. Run zero-shot A/B/C

```bash
python scripts/run_visual_baseline.py \
  --manifest data/splits/val.json \
  --output outputs/predictions/visual_only.jsonl

python scripts/run_risk_conditioned.py \
  --config configs/risk_score.yaml \
  --manifest data/splits/val.json \
  --risk-features data/processed/risk_features.jsonl \
  --output outputs/predictions/risk_score.jsonl

python scripts/run_risk_conditioned.py \
  --config configs/temporal_risk.yaml \
  --manifest data/splits/val.json \
  --risk-features data/processed/risk_features.jsonl \
  --output outputs/predictions/temporal_risk.jsonl
```

### 5. Evaluate and benchmark

```bash
python scripts/evaluate.py \
  --config configs/temporal_risk.yaml \
  --predictions outputs/predictions/temporal_risk.jsonl

python scripts/benchmark_latency.py --config configs/temporal_risk.yaml
pytest -q
```

Metrics are written to
`outputs/experiments/<experiment_id>/metrics.json`; resolved configs are stored
beside them. Large outputs are ignored by Git.

### 6. Launch the research dashboard

```bash
python scripts/launch_demo.py
```

The dashboard can use measured risk/action artifacts or an explicitly marked
synthetic demonstration. Synthetic values are never presented as model output.

## Repository guide

- [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md): scope and evidence gates.
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md): frozen
  experimental procedure.
- [`docs/DATA_SPLIT_REPORT.md`](docs/DATA_SPLIT_REPORT.md): split status and
  leakage rules.
- [`docs/FAILURE_TAXONOMY.md`](docs/FAILURE_TAXONOMY.md): safety-relevant
  failure categories.
- [`docs/RESULTS.md`](docs/RESULTS.md): measured results only.
- [`src/riskvla`](src/riskvla): modular data, risk, VLA, evaluation, training,
  and demo code.

## License and data/model terms

Project code is Apache-2.0; see [`LICENSE`](LICENSE). Third-party datasets and
models retain their own terms. In particular, DRAMA media are governed by the
Honda Research Institute data agreement and must not be committed or
redistributed from this repository.
