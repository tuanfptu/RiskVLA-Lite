"""Primary safety-action constants.

``MANEUVER`` stays in ``EXTENDED_ACTIONS`` so a later lateral-action study can
add it without redefining the primary Nexar benchmark.
"""

from __future__ import annotations

PRIMARY_ACTIONS: tuple[str, ...] = (
    "MAINTAIN",
    "CAUTION",
    "SLOW",
    "BRAKE_OR_STOP",
)

EXTENDED_ACTIONS: tuple[str, ...] = PRIMARY_ACTIONS + ("MANEUVER",)

# Active benchmark action space.
ACTIONS: tuple[str, ...] = PRIMARY_ACTIONS

DEFAULT_SEVERITY: dict[str, int] = {
    "MAINTAIN": 0,
    "CAUTION": 1,
    "SLOW": 2,
    "BRAKE_OR_STOP": 3,
}
