# Failure Taxonomy

Failure analysis is performed on paired A/B/C outputs with the human action,
selected-frame timestamps, risk traces, and access-safe example references.
Media are never redistributed unless their license explicitly permits it.

## Action failures

- **Missed critical action:** target `BRAKE_OR_STOP`, prediction is any less
  urgent action or invalid.
- **Under-reaction:** predicted urgency rank is below target rank.
- **Over-reaction:** predicted urgency rank is above target rank.
- **Unnecessary braking:** prediction is `BRAKE_OR_STOP` while target is not.
- **VLA semantic error:** relevant objects/relations appear to be understood
  incorrectly.
- **VLA action-selection error:** scene interpretation is plausible, but the
  chosen macro-action is inappropriate.
- **Invalid structured output:** malformed schema, unknown action, non-finite
  confidence, or extra non-JSON text.

## Temporal failures

- **Late correct action:** action class is correct, but prediction occurs after
  the required lead-time threshold or event.
- **Post-event leakage:** any selected frame or risk feature occurs at/after a
  known event during an early-action evaluation; this invalidates the sample.
- **Risk detector false positive:** trigger occurs without a supported hazard,
  causing avoidable VLA invocation or intervention.
- **Risk detector false negative:** no useful trigger before a supported
  critical event.
- **Risk detector late activation:** risk rises only after useful action lead
  time has elapsed.

## Perception/context factors

- small or occluded object;
- night, glare, rain, snow, or low contrast;
- multiple simultaneous hazards;
- pedestrian/cyclist ambiguity;
- ego-motion or right-of-way ambiguity;
- insufficient four-frame temporal context;
- frame sampler misses the relevant pre-event evidence.

## Ground-truth/data factors

- ambiguous human action at the observation cutoff;
- action inferred from a collision label, BADAS score, or event time;
- post-cutoff frames seen before the action was assigned;
- source-video leakage across train, validation, and test;
- no verified event time when event lead time is reported;
- no human `actionable_from` when actionable-time error is reported;
- annotation does not distinguish attention, defensive slowing, and strong
  braking.

## Conditioning-specific failures

- VLA ignores numeric risk context;
- high risk anchors the VLA toward braking despite benign visual evidence;
- a noisy slope dominates a more reliable current score;
- treating an in-domain Nexar result as cross-dataset generalization;
- temporal features accidentally include future scores;
- variant prompts differ beyond the declared conditioning block.

## Required failure record

Each saved case includes:

```text
sample_id
source_group (when verified)
human_action
observation_cutoff
variant predictions and validity
selected frame timestamps
event/prediction timestamps (only when verified)
risk feature values and trigger state
failure categories
short observable rationale
media reference and redistribution permission
```

Explanations describe observable evidence; they are not hidden model
chain-of-thought.
