# Limitations

## BADAS on Nexar is in-domain

BADAS-Open is associated with the Nexar dashcam domain. Running it on
`nexar-ai/nexar_collision_prediction` is a useful implementation check: the
risk model is seeing imagery from the domain it was built for.

That setting is not cross-dataset generalization. A gain for temporal risk
conditioning on this subset does not show that the same conditioning would
transfer to another camera, geography, or label definition. This study should
describe the result as in-domain risk conditioning.

## Actions are not in the official dataset

Nexar labels collisions and near-collisions. It does not label `MAINTAIN`,
`CAUTION`, `SLOW`, or `BRAKE_OR_STOP`. Until the human subset exists, no
action metric can be computed. Deriving those names from the collision label,
from BADAS, or from `time_of_event` would manufacture ground truth.

## Lateral maneuvers are out of scope

The primary action list has no `MANEUVER` class. The forward-facing clip does
not include lane geometry, surround perception, or vehicle dynamics. The
extension point remains in the schema so a later study can add it.

## Test clips are a different construction

Official test videos are short and already truncated before the event. They
are not interchangeable with full training videos or with cutoffs we define
ourselves.

## What this repository does not claim

- that temporal risk conditioning improves action selection;
- a first-ever vision-language-action model;
- state-of-the-art driving performance;
- a production driver-assistance system;
- an autonomous-driving controller;
- improved real-world safety.

Those claims need measurements this repository does not yet have.
