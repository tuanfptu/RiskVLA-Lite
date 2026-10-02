# Research Question and Hypotheses

## Primary question

> Can temporal ego-collision risk signals improve the accuracy, timeliness,
> and efficiency of a lightweight vision-language action selector on
> real-world dashcam safety scenarios?

## Primary hypothesis

Temporal BADAS risk conditioning improves safety-action selection relative to
the same Qwen3-VL-2B model using visual evidence alone.

The dataset, human action schema, frame budget, and decoding policy are held
constant. Variant C adds current risk, risk slope, and recent peak. Variant A
sees the same frames and no risk features. The hypothesis is untested.

## Secondary hypotheses

1. Current risk alone (B) provides less useful signal than temporal risk (C).
2. A validation-tuned risk or risk-plus-slope trigger lowers VLA invocation
   rate while preserving `BRAKE_OR_STOP` recall relative to always-on
   inference.
3. Where human `actionable_from` values exist, temporal conditioning reduces
   actionable-time error without using post-cutoff frames. Event lead time
   remains a separate measurement.

## Null and adverse outcomes

- The primary null is no stable difference between A and C.
- An accuracy gain with worse `BRAKE_OR_STOP` recall is not a safety
  improvement.
- A gain caused by video leakage, different frames, prompt drift, or test-set
  threshold tuning is invalid.
- Better classification with unacceptable latency is a mixed result.
- An in-domain Nexar result is not evidence of cross-dataset generalization.

No novelty, production, or safety-benefit claim is made before the controlled
ablation has measured evidence.
