# Results

## Research conclusion: NOT YET MEASURED

There is currently insufficient evidence to answer whether temporal risk
conditioning improves VLA safety-action selection.

No accuracy, F1, recall, timing, latency, FPS, memory, or safety-improvement
number has been filled with a placeholder or synthetic value.

## Primary ablation

| Variant | Visual | Current Risk | Risk Trend | Macro-F1 | Critical Recall | Under-Reaction | Over-Reaction | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A Visual-only | ✓ |  |  | — | — | — | — | — |
| B Risk-score | ✓ | ✓ |  | — | — | — | — | — |
| C Temporal-risk | ✓ | ✓ | ✓ | — | — | — | — | — |

`—` means not measured, not zero.

## Trigger efficiency

| Policy | VLA Invocation Rate | Critical Recall | Macro-F1 | Avg End-to-End Latency |
|---|---:|---:|---:|---:|
| Always-on | — | — | — | — |
| Risk threshold | — | — | — | — |
| Risk + slope | — | — | — | — |

## Blocking evidence

- Public DRAMA-X annotations do not include media or an authoritative source
  group for leakage-safe splitting.
- They do not contain event timestamps, so action lead time is unsupported.
- BADAS weights require approved gated access.
- The current development machine has no NVIDIA runtime.

## Result publication rule

This document may be populated only from committed machine-readable experiment
records. Each table update must cite experiment IDs, Git commit, split/mapping
versions, hardware, precision, and confidence/instability analysis. A failed or
negative ablation remains in the table.
