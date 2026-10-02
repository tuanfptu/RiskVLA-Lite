"""Seed and environment helpers used by experiment entry points."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_reproducible_seed(seed: int = 42, *, deterministic_torch: bool = False) -> dict[str, Any]:
    """Seed available RNGs and return a record of applied settings.

    Torch is optional so annotation tooling and unit tests do not require the
    multi-gigabyte inference dependency.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    record: dict[str, Any] = {
        "seed": seed,
        "python": True,
        "numpy": True,
        "torch": False,
        "deterministic_torch": False,
    }
    try:
        import torch
    except ImportError:
        return record

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
    record["torch"] = True
    record["deterministic_torch"] = deterministic_torch
    return record
