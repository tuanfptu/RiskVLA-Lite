# Results

## Research conclusion: NOT MEASURED

There is no evidence yet that temporal risk conditioning improves safety-action
selection. The hypothesis is unchanged and untested.

No accuracy, F1, recall, timing, latency, memory, or safety-improvement number
has been filled with a placeholder.

## Primary ablation

| Variant | Visual | Current risk | Temporal risk | Macro-F1 | BRAKE_OR_STOP recall | Under-reaction | Over-reaction | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A Visual only | yes |  |  | — | — | — | — | — |
| B Current risk | yes | yes |  | — | — | — | — | — |
| C Temporal risk | yes | yes | yes | — | — | — | — | — |

`—` means not measured.

## Trigger comparison

| Policy | VLA invocation rate | BRAKE_OR_STOP recall | Macro-F1 | End-to-end latency |
|---|---:|---:|---:|---:|
| Always-on | — | — | — | — |
| Risk threshold | — | — | — | — |
| Risk + slope | — | — | — | — |

## Timing

Event lead time and actionable-time error are defined and implemented. Both
are **NOT MEASURED** because there are no human action timestamps and no model
predictions.

## Why nothing is filled in

- Nexar media are not downloaded in this environment.
- Human action annotations have not been created.
- The action split has not been created.
- BADAS and Qwen have not been run on a GPU.

## Publication rule

Fill a cell only from a committed experiment record. Cite the experiment id,
Git commit, split version, action-schema version, hardware, and precision.
A negative result stays in the table.
