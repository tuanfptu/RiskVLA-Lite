"""Small classifier for explicitly extracted frozen Qwen representations."""

from __future__ import annotations

from typing import Any


def build_action_head(
    input_dim: int = 2048,
    *,
    hidden_dim: int = 256,
    num_actions: int = 5,
    dropout: float = 0.1,
) -> Any:
    """Build a lightweight Torch head without making Torch a core dependency."""
    if input_dim <= 0 or hidden_dim <= 0 or num_actions <= 1:
        raise ValueError("Invalid action-head dimensions")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    try:
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Torch is required for the optional action head") from exc
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_actions),
    )
