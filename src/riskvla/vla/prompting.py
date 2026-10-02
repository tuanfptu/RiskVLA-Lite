"""Controlled prompts whose primary endpoint is a discrete action."""

from __future__ import annotations

from riskvla.risk.features import RiskFeatures
from riskvla.vla.conditioning import ConditioningVariant, build_conditioning_context

ACTION_DEFINITIONS = """\
- MAINTAIN: no meaningful defensive intervention is required at the observation cutoff
- CAUTION: a potential hazard deserves increased attention, but immediate deceleration or braking is not yet justified
- SLOW: defensive deceleration is appropriate, but emergency or strong braking is not yet justified
- BRAKE_OR_STOP: immediate substantial braking or stopping is justified by the visible safety situation"""


def build_action_prompt(
    *,
    variant: ConditioningVariant | str,
    risk_features: RiskFeatures | None = None,
    diagnostic: bool = False,
) -> str:
    context = build_conditioning_context(variant, risk_features)
    context_block = (
        ""
        if context is None
        else (
            "\nA frozen upstream risk detector supplied these causal values. "
            "They may be noisy; use them as context, not as an action label:\n"
            f"{context}\n"
        )
    )
    if diagnostic:
        schema = (
            '{"hazard_type":"short observable category",'
            '"ego_relation":"short spatial relation",'
            '"action":"ONE_ALLOWED_ACTION","confidence":0.0}'
        )
    else:
        schema = '{"action":"ONE_ALLOWED_ACTION","confidence":0.0}'

    return f"""\
You are selecting one research-only ego-vehicle safety recommendation from
ordered dashcam frames. Use only evidence visible by the final supplied frame.
Do not infer future frames and do not provide chain-of-thought.

Allowed actions:
{ACTION_DEFINITIONS}
{context_block}
Return exactly one JSON object, with no Markdown or extra text:
{schema}

`action` must exactly match one allowed action. `confidence` must be a number
from 0 to 1. Select the least disruptive action that adequately addresses the
visible situation."""
