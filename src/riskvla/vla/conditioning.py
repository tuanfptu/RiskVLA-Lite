"""Interpretable A/B/C risk-conditioning blocks."""

from __future__ import annotations

from enum import Enum

from riskvla.risk.features import RiskFeatures


class ConditioningVariant(str, Enum):
    VISUAL_ONLY = "visual_only"
    VISUAL_PLUS_CURRENT_RISK = "visual_plus_current_risk"
    VISUAL_PLUS_TEMPORAL_RISK = "visual_plus_temporal_risk"


def build_conditioning_context(
    variant: ConditioningVariant | str,
    risk: RiskFeatures | None,
) -> str | None:
    selected = ConditioningVariant(variant)
    if selected is ConditioningVariant.VISUAL_ONLY:
        if risk is not None:
            raise ValueError("Visual-only variant must not receive risk features")
        return None
    if risk is None:
        raise ValueError(f"{selected.value} requires risk features")
    if selected is ConditioningVariant.VISUAL_PLUS_CURRENT_RISK:
        return f"Current upstream risk score R_t: {risk.current_risk:.3f}"
    return "\n".join(
        [
            f"Current upstream risk score R_t: {risk.current_risk:.3f}",
            f"Recent risk slope per second: {risk.risk_slope:+.3f}",
            f"Recent maximum risk: {risk.recent_peak:.3f}",
        ]
    )
