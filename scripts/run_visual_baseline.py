#!/usr/bin/env python3
"""Run zero-shot variant A with no risk context."""

from __future__ import annotations

import argparse
from pathlib import Path

from _inference import run_zero_shot

from riskvla.vla.conditioning import ConditioningVariant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/visual_only.yaml")
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run_zero_shot(
        config_path=args.config,
        manifest_path=args.manifest,
        output_path=args.output,
        variant=ConditioningVariant.VISUAL_ONLY,
        media_root=args.media_root,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
