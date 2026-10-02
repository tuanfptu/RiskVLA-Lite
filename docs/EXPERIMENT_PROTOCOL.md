# Experiment Protocol

## Order

1. Download Nexar on persistent storage. Do not download it in the cloud
   development environment.
2. Annotate 300–500 clips with the human protocol. Do not auto-label.
3. Freeze the action schema, prompts, and the video-level split.
4. Run zero-shot A, B, and C on the same validation rows with four frames.
5. Select trigger thresholds on validation only.
6. Consider eight frames only if the four-frame result justifies the extra
   visual context.
7. Only then decide whether a frozen action head or LoRA is justified.
8. Freeze the chosen configuration and evaluate test once.

LoRA status remains `NOT_STARTED` until step 7 has evidence. BADAS stays
frozen throughout.

Every run stores a resolved config and metrics record under
`outputs/experiments/<experiment_id>/`.

## Controls

Held constant across A/B/C:

- Qwen revision and precision;
- eligible sample ids and split;
- the same pre-cutoff frames;
- prompt text except the declared conditioning block;
- greedy decoding (`do_sample=false`), one beam, and a 64-token ceiling;
- the strict four-class parser;
- metric definitions;
- seed 42.

Variant A must not receive risk through filenames, metadata, frame timing
chosen from future risk, or prompt text. Risk-centered frame selection is a
separate experiment.

## Output

```json
{"action": "SLOW", "confidence": 0.84}
```

Allowed actions are `MAINTAIN`, `CAUTION`, `SLOW`, and `BRAKE_OR_STOP`.
Confidence is finite and inside `[0, 1]`. Extra prose, unknown labels
including `MANEUVER`, malformed JSON, and invalid confidence stay invalid.

## Risk features at cutoff t

- `current_risk`: latest finite score at or before `t`
- `risk_slope`: least-squares slope over the trailing window
- `recent_peak`: maximum finite score in the trailing peak window
- diagnostics: moving mean and volatility
- `trigger_timestamp`: first time the selected policy fires at or before `t`

No future score enters the features. Warm-up `NaN` values stay invalid.

## Frames

Default: 4 frames with timestamps at or before the observation cutoff and,
when an event exists, strictly before `time_of_event`. Record the timestamps.
Eight frames are not the primary setting.

## Metrics

Report accuracy, macro-F1, per-class precision, recall, and F1, plus:

- `BRAKE_OR_STOP` recall
- under-reaction and over-reaction
- unnecessary braking
- BADAS, VLA, and end-to-end latency
- peak VRAM
- VLA invocation rate

When the timestamps exist, also report event lead time and actionable-time
error as separate quantities:

```text
event_lead_time = time_of_event - predicted_action_time
actionable_time_error = predicted_action_time - human_actionable_time
```

`actionable_from` is a human field. It is not a BADAS timestamp.

## Thresholds

Candidate thresholds are declared before the validation search. Selection
maximizes validation critical recall subject to an invocation budget. Ties
prefer the lower invocation rate, then the higher threshold. Test rows are
not used to pick thresholds.

## Statistics

Bootstrap over source videos, not over correlated cutoffs from the same
video. If too few videos exist, report that instability instead of a
significance claim.

## Optional adaptation

A frozen action head or Qwen LoRA starts only after zero-shot A/B/C
validation. Language-attention LoRA targets remain `q_proj`, `k_proj`,
`v_proj`, and `o_proj`, checked against the loaded module names. The visual
encoder and BADAS stay frozen.
