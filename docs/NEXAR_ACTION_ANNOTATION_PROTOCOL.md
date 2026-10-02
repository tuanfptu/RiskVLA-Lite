# Nexar Action Annotation Protocol

Status: schema defined; no annotations have been created.

Target size for the first subset: 300–500 clips. The sampler may propose
candidates. It must not assign actions.

## Decision rule

The action is the least disruptive defensive response justified by what is
visible at the observation cutoff.

Labels use only visual evidence available at or before `observation_cutoff`.
The annotator must not inspect post-cutoff frames before assigning the action.
That rule prevents the eventual event from leaking into the label.

Do not assign the action from:

- the Nexar collision label;
- `time_of_event` or `time_of_alert`;
- BADAS risk scores or triggers;
- Qwen or any other model output;
- a time-to-collision threshold.

Those values may be stored as source metadata. They are not the label.

## Actions

### MAINTAIN

No meaningful defensive intervention is required at the observation cutoff.
The visible scene does not justify a change in speed or an emergency response.

### CAUTION

A potential hazard deserves increased attention, but immediate deceleration
or braking is not yet justified.

### SLOW

Defensive deceleration is appropriate, but emergency or strong braking is not
yet justified.

### BRAKE_OR_STOP

Immediate substantial braking or stopping is justified by the visible safety
situation.

`MANEUVER` is not in this benchmark. A single forward-facing clip does not
include the lane geometry, surround state, or vehicle dynamics needed to
grade a specific lateral maneuver. The code keeps the name available for a
later extension.

## Observation cutoff

For a positive clip with a verified `time_of_event`, candidate cutoffs are:

```text
time_of_event - 3.0 s
time_of_event - 2.0 s
time_of_event - 1.0 s
```

The offsets are configuration, not a requirement that every experiment use
all three. A cutoff that would fall before the start of the clip is omitted.

For a negative clip, store an explicit observation timestamp. Choose times
that are temporally comparable to the positive cutoffs, such as the same
anchor times when they fall inside the clip. Do not invent an event, an
alert, or a hazard in order to create that timestamp.

Every frame given to Qwen must satisfy `frame_time <= observation_cutoff`.
When `time_of_event` exists, every frame must also satisfy
`frame_time < time_of_event`.

The annotation interface should reveal frames only up to the cutoff. It
should not reveal the event time as a hint while the annotator is choosing
the action.

## Other fields

| Field | Rule |
|---|---|
| `actionable_from` | Human timestamp when an intervention first becomes justified, or null. It must not be derived from BADAS. It cannot be after the cutoff or at/after `time_of_event`. |
| `ego_relevance` | `none` if the situation does not involve the ego path; `indirect` if it is nearby but not clearly on that path; `direct` if it clearly affects the ego path. |
| `confidence` | Integer 1–5. 1 means very uncertain; 5 means the visible evidence strongly supports one action. |
| `ambiguous` | True when more than one adjacent action remains plausible after looking only at pre-cutoff frames. |
| `annotator` | Person who assigned the action. |
| `notes` | Free text about visible evidence. Not a substitute for the action. |
| `nexar_metadata` | Optional preserved official metadata. Ignored when the action is read. |

## Sampling intent

The first subset should be able to include:

- positive collision or near-collision clips;
- negative normal-driving clips;
- early-hazard views, by labeling earlier candidate cutoffs;
- normal scenes that are easy to mistake for hazards;
- urban and highway scenes;
- more than one weather and light condition when the metadata contains them.

The automatic proposal stratifies by collision label and scene and copies
weather and light onto the candidate row. It does not decide which scenes are
false alarms, and it does not label early hazards. Those two groups are
manual enrichment after a person, or a later reviewed BADAS diagnostic, has
looked at the clips.

## Recorded schema

`data/annotations/nexar_action_schema.json`
