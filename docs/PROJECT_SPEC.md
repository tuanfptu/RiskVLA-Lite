# RiskVLA-Lite Project Specification

Status: infrastructure-ready; benchmark execution blocked on external media and
model access as described below.

## Objective

RiskVLA-Lite tests one falsifiable question:

> Does conditioning a lightweight visual-language action selector on temporal
> collision-risk signals improve safety-action classification, action timing,
> or selective-invocation efficiency beyond visual input alone?

The system recommends a discrete ego-vehicle safety action. It is not a
controller, planner, certified driver-assistance system, or safety case.

## System boundary

```text
dashcam media
  -> frozen RiskProvider
  -> normalized risk sequence and temporal features
  -> leakage-safe frame selection
  -> replaceable VLA action selector
  -> one validated macro-action
```

The upstream detector answers *when risk is emerging*. The VLA answers *what
action is appropriate*. No actuator interface is included.

## Primary implementations

- Risk provider: BADAS-Open, frozen, loaded only from an official checkpoint.
- VLA: `Qwen/Qwen3-VL-2B-Instruct`.
- Benchmark annotations: the canonical `mgod96/DRAMA-X` JSONL.
- Target hardware: one NVIDIA RTX 3090 (24 GB); normal BF16/FP16 is measured
  before any quantized configuration.

Both model integrations sit behind project-owned interfaces. Mock providers
exist only for deterministic tests and infrastructure demonstrations; mock
outputs must never be reported as benchmark results.

## Fixed action space

The initial macro-actions are:

1. `MAINTAIN`
2. `CAUTION`
3. `SLOW`
4. `BRAKE_OR_STOP`
5. `MANEUVER`

`configs/action_mapping.yaml` is versioned and frozen before model evaluation.
Every processed record keeps its exact native `suggested_action`. The native
`N/A` label is explicitly excluded rather than being reinterpreted as
`MAINTAIN`.

## Primary ablation

| Variant | Visual context | Current risk | Temporal risk |
|---|---:|---:|---:|
| A — visual-only | yes | no | no |
| B — current-risk | yes | `R_t` | no |
| C — temporal-risk | yes | `R_t` | slope and recent peak |

All variants use the same eligible examples, split version, sampled-frame
budget, decoding policy, and output parser. Greedy decoding and strict JSON
validation are used. Invalid model output remains invalid and is counted.

An optional frozen action head or LoRA adaptation is out of the primary scope
until A–C have valid validation results. LoRA never updates BADAS or the visual
encoder initially.

## Data and leakage policy

- Seed: 42.
- Split unit: verified source video, clip, or source sequence.
- Individual image IDs are never treated as independent split groups unless
  the dataset owner documents that they are independent.
- Frames selected for an early-action prediction must be at or before the
  prediction cutoff and strictly before a known hazardous-event time.
- Test data are evaluated once after mapping, prompts, thresholds, and configs
  are frozen from training/validation work.

The public DRAMA-X annotation currently has empty media fields and no
authoritative group or event-time field. Consequently, committed split
manifests are blocked-state manifests, not invented random splits. The split
tool refuses to proceed without defensible grouping metadata.

## Metrics

Primary classification metrics:

- accuracy;
- macro-F1;
- per-class precision, recall, and F1;
- confusion matrix;
- invalid-output rate.

Safety metrics:

- `BRAKE_OR_STOP` recall;
- severity-based under-reaction rate;
- severity-based over-reaction rate;
- unnecessary-braking rate.

Efficiency metrics:

- risk-provider, VLA, and end-to-end latency;
- peak GPU memory;
- VLA invocation count/rate.

Temporal metrics are reported only when a verified event timestamp and
prediction timestamp exist. Action lead time is
`event_time - prediction_time`; positive values are pre-event.

## Trigger policy

Two validation-tuned policies are supported:

```text
current:   R_t > risk_threshold
temporal:  R_t > risk_threshold OR slope > slope_threshold
```

Thresholds are configuration values. They may be selected on validation data
but never optimized on test data.

## Experiment record

Each experiment directory contains the resolved configuration and a metrics
record with:

- experiment ID and UTC timestamp;
- Git commit and dirty-state flag;
- split and action-mapping versions;
- model and risk-provider versions;
- seed, GPU, precision;
- metrics, latency, and peak VRAM;
- artifact status (`MEASURED`, `BLOCKED`, or `FAILED`).

## Evidence gates

A component is called working only after a meaningful runtime check.

- BADAS: official checkpoint loads, one real video runs, and a bounded finite
  post-warm-up score sequence is produced.
- Qwen: official weights load, four real frames produce parseable structured
  output, and device/latency/VRAM are recorded.
- DRAMA-X benchmark: media resolve, split grouping is defensible, and all
  eligible examples map without unknown labels.
- Early action: verified event timestamps exist and all selected frames precede
  each event.

As of 2026-10-02, none of these model/data runtime gates has been claimed
without the required external assets. Unit-tested infrastructure is reported
separately from measured research evidence.

## Kill criteria

The hypothesis is unsupported when temporal conditioning fails to improve
stable safety/action metrics, introduces leakage, lowers critical recall
materially, activates too late, has impractical latency, or relies on labels
too ambiguous for the proposed endpoint. Negative and mixed outcomes are valid
results and must be reported without reinterpretation.
