#!/usr/bin/env python3
"""Run zero-shot variant B or C with precomputed causal risk features."""

from __future__ import annotations

import argparse
from pathlib import Path

from _inference import run_zero_shot

from riskvla.config import load_config
from riskvla.vla.conditioning import ConditioningVariant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--risk-features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow-synthetic-risk",
        action="store_true",
        help="Infrastructure testing only; output cannot be benchmark evidence.",
    )
    args = parser.parse_args()
    variant = ConditioningVariant(load_config(args.config)["vla"]["variant"])
    if variant is ConditioningVariant.VISUAL_ONLY:
        raise ValueError("Use run_visual_baseline.py for the visual-only variant")
    run_zero_shot(
        config_path=args.config,
        manifest_path=args.manifest,
        output_path=args.output,
        variant=variant,
        risk_features_path=args.risk_features,
        media_root=args.media_root,
        limit=args.limit,
        allow_synthetic_risk=args.allow_synthetic_risk,
    )


if __name__ == "__main__":
    main()
