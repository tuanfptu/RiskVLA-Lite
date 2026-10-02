# Experiment Protocol

## Governance

1. Verify media access and source grouping.
2. Freeze action mapping and split manifests.
3. Run zero-shot A/B/C on the same validation rows with four frames.
4. Select trigger thresholds on validation only.
5. Repeat with eight frames only if four-frame evidence motivates it.
6. Decide whether a frozen head or LoRA is justified.
7. Freeze configs and evaluate test once.

Every run stores a resolved config and metrics record under
`outputs/experiments/<experiment_id>/`.

## Controls

Held constant across A/B/C:

- Qwen model revision and precision;
- eligible sample IDs and split;
- image preprocessing and frame indices;
- prompt wording except the declared conditioning block;
- generation settings (`do_sample=false`, one beam, 64-token ceiling);
- strict output parser;
- action mapping and metric definitions;
- random seed 42.

BADAS remains frozen. Variant A does not receive risk values indirectly through
filenames, metadata, frame timing selected from future risk, or prompt text.
For the primary conditioning ablation, all variants use the same frame set.
Risk-centered frame selection is a separate ablation because it changes visual
evidence.

## Output schema

Primary output:

```json
{"action": "SLOW", "confidence": 0.87}
```

Allowed actions are the five frozen macro-actions. Confidence must be finite
and within `[0, 1]`. Extra prose, unknown labels, malformed JSON, duplicate
objects, and invalid confidence values are invalid outputs. They are not
coerced and contribute false negatives to class metrics.

## Risk features

At prediction cutoff `t`:

- `current_risk`: latest finite score at or before `t`;
- `risk_slope`: least-squares slope per second over the configured trailing
  window;
- `recent_peak`: maximum finite score in the trailing peak window;
- diagnostic-only `risk_mean` and `risk_volatility`;
- `trigger_timestamp`: first threshold/slope policy activation.

No future score may enter features. Warm-up `NaN` values remain invalid.

## Frame policy

The default uses four uniformly selected frames from an allowed pre-cutoff
window. When event time is known, every frame timestamp must be strictly less
than the event. A risk-centered policy may center a pre-trigger window on a
trigger at or before the prediction cutoff; it may never extend into future or
post-event frames.

## Metrics

Report accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, and
invalid-output rate. Also report:

- critical recall for `BRAKE_OR_STOP`;
- under-reaction (predicted severity lower than target);
- over-reaction (predicted severity higher than target);
- unnecessary braking (`BRAKE_OR_STOP` predicted for a non-brake target);
- risk, VLA, and end-to-end latency;
- peak CUDA allocation;
- invocation count/rate.

`SLOW` and `MANEUVER` have equal urgency rank but are still distinct classes;
confusing them is a semantic action error, not an under/over-reaction.

Lead-time statistics are produced only for rows with verified event and
prediction timestamps:

```text
lead_time = event_time - prediction_time
```

The report includes mean, median, distribution data, and fractions at least
0.5, 1.0, and 2.0 seconds early.

## Threshold selection

Candidate trigger thresholds are specified before validation search. Selection
maximizes validation critical recall subject to an invocation-rate budget; ties
prefer lower invocation rate, then the higher threshold. Test data are never
used to select thresholds.

## Statistical interpretation

Use paired bootstrap resampling over source groups, not individual correlated
frames, for confidence intervals on A–C differences. If too few independent
groups exist, report instability and raw paired differences rather than a
misleading significance claim.

## Optional adaptation gate

A frozen action head is attempted only when a stable hidden-state extraction
path and enough grouped training data exist. LoRA is attempted only after
zero-shot A–C validation, with a predeclared rationale. Starting targets are
language-attention `q_proj`, `k_proj`, `v_proj`, and `o_proj`; actual names are
validated at runtime. The visual encoder and BADAS remain frozen initially.

## Reproducibility

Record Python, NumPy, and Torch seeds; software/model revisions; Git commit and
dirty state; GPU identity; precision; deterministic settings; and known
nondeterminism. Fused GPU kernels may not be bitwise deterministic even when
seeds are fixed; this is reported rather than hidden.
