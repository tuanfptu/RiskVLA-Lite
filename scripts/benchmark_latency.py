#!/usr/bin/env python3
"""Measure Qwen and optional BADAS-to-Qwen latency on the actual target host."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from riskvla.config import load_config
from riskvla.pipeline import RiskVLAPipeline
from riskvla.risk.badas import BADASRiskProvider
from riskvla.vla.conditioning import ConditioningVariant
from riskvla.vla.qwen import QwenActionSelector, QwenConfig


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(values) - 1, int(0.95 * len(values) + 0.9999) - 1))
    return {
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": ordered[p95_index],
        "minimum_seconds": ordered[0],
        "maximum_seconds": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--frames", required=True, nargs="+", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/latency_benchmark.json")
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("warmup must be non-negative and runs must be positive")

    config = load_config(args.config)
    variant = ConditioningVariant(config["vla"]["variant"])
    selector = QwenActionSelector(
        QwenConfig(
            model_id=config["vla"]["model_id"],
            revision=str(config["vla"].get("model_revision", "main")),
            device=config["vla"]["device"],
            precision=config["vla"]["precision"],
            attention_implementation=config["vla"]["attention_implementation"],
            max_new_tokens=int(config["vla"]["max_new_tokens"]),
            max_pixels_per_frame=config["vision"]["max_pixels_per_frame"],
        )
    )
    risk_provider = None
    if variant is not ConditioningVariant.VISUAL_ONLY:
        if args.video is None:
            raise ValueError("Risk-conditioned end-to-end timing requires --video")
        risk_provider = BADASRiskProvider(
            checkpoint_path=args.checkpoint,
            device=config["risk"].get("device", config["vla"]["device"]),
        )
    pipeline = RiskVLAPipeline(
        selector=selector,
        variant=variant,
        risk_provider=risk_provider,
    )

    def invoke() -> Any:
        return pipeline.predict(
            frames=args.frames,
            video_path=args.video,
        )

    for _ in range(args.warmup):
        invoke()
    results = [invoke() for _ in range(args.runs)]
    valid_count = sum(result.valid for result in results)
    risk_values = [
        result.risk_latency_seconds
        for result in results
        if result.risk_latency_seconds is not None
    ]
    report = {
        "status": "MEASURED",
        "config": str(args.config),
        "variant": variant.value,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "valid_output_rate": valid_count / args.runs,
        "vla_latency": summarize(
            [result.vla_latency_seconds for result in results]
        ),
        "risk_latency": summarize(risk_values) if risk_values else None,
        "end_to_end_latency": summarize(
            [result.end_to_end_latency_seconds for result in results]
        ),
        "peak_vram_bytes": max(
            (
                result.peak_vram_bytes
                for result in results
                if result.peak_vram_bytes is not None
            ),
            default=None,
        ),
        "device": results[0].metadata["selector"]["device"],
        "precision": config["vla"]["precision"],
        "frame_count": len(args.frames),
        "note": "Numbers apply only to this recorded machine/configuration.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
