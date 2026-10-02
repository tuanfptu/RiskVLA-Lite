# Research Question and Hypotheses

## Primary question

> Does temporal collision-risk information provide measurable value beyond
> visual-language understanding alone for safety-critical action selection?

## Primary hypothesis

With the dataset, mapping, frame budget, and decoding policy held constant,
variant C (visual plus current risk, slope, and recent peak) will improve
macro-F1 and/or `BRAKE_OR_STOP` recall relative to variant A (visual-only)
without a material increase in over-reaction.

## Secondary hypotheses

1. Current risk alone (B) provides less useful signal than temporal risk (C).
2. A validation-tuned risk-or-slope trigger lowers VLA invocation rate while
   preserving critical-action recall relative to always-on inference.
3. Where verified event times exist, temporal conditioning produces a larger
   positive action lead time without relying on post-event frames.

## Null and adverse outcomes

- The primary null is no stable difference between A and C.
- An accuracy gain with worse critical recall is not a safety improvement.
- A gain caused by group leakage, different eligible examples, prompt drift,
  or test-set threshold tuning is invalid.
- Better classification with unacceptable latency is a mixed result.
- If BADAS scores have negligible relation to action targets, the proposed
  conditioning signal is unsupported for this benchmark.

No novelty or safety-benefit claim is made before the controlled ablation has
measured evidence.
