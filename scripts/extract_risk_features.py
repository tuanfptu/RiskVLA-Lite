#!/usr/bin/env python3
"""Extract causal normalized risk features for a ready split manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from riskvla.io import read_json, write_jsonl
from riskvla.risk.badas import BADASRiskProvider
from riskvla.risk.base import RiskProvider, RiskSequence
from riskvla.risk.features import TriggerConfig, extract_risk_features
from riskvla.risk.mock import MockRiskProvider


def build_provider(args: argparse.Namespace) -> RiskProvider:
    if args.provider == "mock":
        return MockRiskProvider(require_media=True)
    return BADASRiskProvider(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", choices=["badas", "mock"], default="badas")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slope-window-seconds", type=float, default=2.0)
    parser.add_argument("--peak-window-seconds", type=float, default=2.0)
    parser.add_argument(
        "--trigger-policy",
        choices=["always", "risk_threshold", "risk_or_slope"],
        default="risk_or_slope",
    )
    parser.add_argument("--risk-threshold", type=float, default=0.8)
    parser.add_argument("--slope-threshold", type=float, default=0.15)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    if manifest.get("status") != "READY":
        raise RuntimeError(
            f"Split manifest is not READY: {manifest.get('status', 'missing status')}"
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("Ready manifest has no samples")

    provider = build_provider(args)
    trigger = TriggerConfig(
        policy=args.trigger_policy,
        risk_threshold=args.risk_threshold,
        slope_threshold=args.slope_threshold,
    )
    sequence_cache: dict[str, RiskSequence] = {}
    output: list[dict[str, Any]] = []
    for sample in samples:
        video_path = str(sample.get("video_path", "")).strip()
        if not video_path:
            raise RuntimeError(
                f"Sample {sample.get('id')} has no video path; risk extraction is blocked"
            )
        if video_path not in sequence_cache:
            sequence_cache[video_path] = provider.predict(video_path)
        sequence = sequence_cache[video_path]
        prediction_time = sample.get("prediction_time")
        features = extract_risk_features(
            sequence,
            prediction_timestamp=(
                None if prediction_time is None else float(prediction_time)
            ),
            slope_window_seconds=args.slope_window_seconds,
            peak_window_seconds=args.peak_window_seconds,
            trigger=trigger,
        )
        output.append(
            {
                "id": sample["id"],
                "source_group": sample["group_id"],
                "provider": provider.describe(),
                "synthetic": bool(sequence.metadata.get("synthetic", False)),
                "features": features.to_dict(),
                "risk_runtime": {
                    key: sequence.metadata.get(key)
                    for key in (
                        "model_revision",
                        "source_revision",
                        "device_requested",
                        "latency_seconds",
                        "peak_vram_bytes",
                        "score_semantics",
                    )
                },
            }
        )
    write_jsonl(args.output, output)
    print(
        f"Wrote {len(output)} samples from {len(sequence_cache)} videos to {args.output}"
    )
    if args.provider == "mock":
        print("WARNING: output is synthetic and is not benchmark evidence.")


if __name__ == "__main__":
    main()
