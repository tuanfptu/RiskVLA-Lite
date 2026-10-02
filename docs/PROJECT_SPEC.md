# RiskVLA-Lite Project Specification

Status: Nexar infrastructure is in place. Media, human actions, and GPU
measurements are not.

## Question

> Can temporal ego-collision risk signals improve the accuracy, timeliness,
> and efficiency of a lightweight vision-language action selector on
> real-world dashcam safety scenarios?

Hypothesis to test, not a result:

> Temporal BADAS risk conditioning improves safety-action selection relative
> to the same Qwen3-VL-2B model using visual evidence alone.

The system recommends one discrete safety action. It is not a controller,
planner, certified driver-assistance system, or safety case.

## Pipeline

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

BADAS answers when collision risk is emerging. The action selector answers
what the ego vehicle should do. No actuator interface is included.

## Primary pieces

- Dataset: `nexar-ai/nexar_collision_prediction`, revision
  `7535d0656dac31d7da2846913bb69c6331d2a70a`.
- Risk provider: BADAS-Open, frozen, official checkpoint only.
- VLA: `Qwen/Qwen3-VL-2B-Instruct`.
- Actions: human labels on a 300–500 clip subset.
- Hardware target: one NVIDIA RTX 3090. This development environment has no
  CUDA GPU, so model runs wait for that machine.

Mock providers exist for tests. Their outputs are not benchmark results.

## Action space

1. `MAINTAIN`
2. `CAUTION`
3. `SLOW`
4. `BRAKE_OR_STOP`

`configs/nexar_actions.yaml` is the active schema. `MANEUVER` is listed only
as a future extension. The old DRAMA-X native-label file is under
`configs/legacy/` and must not be used as Nexar ground truth.

## Primary ablation

| Variant | Visual context | Current risk | Temporal risk |
|---|---:|---:|---:|
| A — visual only | yes | no | no |
| B — current risk | yes | `R_t` | no |
| C — temporal risk | yes | `R_t` | slope and recent peak |

A, B, and C use the same frames, eligible ids, decoding policy, and parser.
The default frame count is 4. An eight-frame run is a later experiment.
Risk-centered sampling is separate because it changes the images.

Zero-shot A/B/C comes before a frozen action head or LoRA. LoRA status is
`NOT_STARTED`. BADAS is never fine-tuned.

## Cutoffs and leakage

Positive candidate cutoffs are `time_of_event` minus a configured offset.
The default candidates are 3, 2, and 1 seconds. An experiment may use a
subset. Negative clips store an explicit observation time and do not receive
a fabricated event.

Frames must satisfy `frame_time <= observation_cutoff`. When an event time
exists, they must also satisfy `frame_time < time_of_event`. Risk features
at cutoff `t` may use only scores at or before `t`.

## Splits

Do not reuse the official Nexar train/test division as the action split.
After manual labels exist, assign source videos with seed 42 to 70% train,
15% validation, and 15% test. All cutoffs from one video share a split.
Thresholds are selected on validation only.

## Metrics

Classification: accuracy, macro-F1, and per-class precision, recall, and F1.

Safety: `BRAKE_OR_STOP` recall, under-reaction, over-reaction, and
unnecessary braking.

Timing, only when the required human or event timestamps exist:

- event lead time = `time_of_event - predicted_action_time`
- actionable-time error = `predicted_action_time - human_actionable_time`

These two numbers are not interchangeable.

Efficiency: BADAS latency, VLA latency, end-to-end latency, peak VRAM, and
VLA invocation rate under always-on, risk-threshold, and risk-plus-slope
invocation.

## Selective invocation

Compare always-on VLA, a risk-threshold trigger, and a risk-plus-slope
trigger. Choose thresholds on validation. Do not tune them on test.

## Evidence gates

- Nexar media: downloaded under the official license, outside this cloud
  environment.
- Actions: a person fills the schema. No automatic action labels.
- BADAS: official checkpoint loads on the GPU machine and one real video
  yields a finite post-warm-up sequence.
- Qwen: official weights produce one parseable action object from four real
  pre-cutoff frames.
- A/B/C: measured on the human subset after the split exists.

## Contribution under test

1. Risk-conditioned lightweight safety-action selection.
2. Selective, risk-triggered VLA invocation.
3. Early-action evaluation on human-annotated Nexar clips.

No novelty, safety, or production claim is made before those measurements
exist.
