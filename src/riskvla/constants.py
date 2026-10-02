"""Shared action constants."""

from __future__ import annotations

ACTIONS: tuple[str, ...] = (
    "MAINTAIN",
    "CAUTION",
    "SLOW",
    "BRAKE_OR_STOP",
    "MANEUVER",
)

DEFAULT_SEVERITY: dict[str, int] = {
    "MAINTAIN": 0,
    "CAUTION": 1,
    "SLOW": 2,
    "MANEUVER": 2,
    "BRAKE_OR_STOP": 3,
}
