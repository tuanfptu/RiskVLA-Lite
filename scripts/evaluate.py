#!/usr/bin/env python3
"""Evaluate structured predictions and write a governed experiment record."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from riskvla.config import load_config
from riskvla.data.labels import ActionMapping
from riskvla.evaluation.experiment import write_experiment
from riskvla.evaluation.latency import LatencyMeasurement, summarize_latency
from riskvla.evaluation.metrics import evaluate_actions
from riskvla.evaluation.timing import action_lead_time_metrics
from riskvla.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/experiments")
    )
    parser.add_argument("--experiment-id")
    parser.add_argument("--allow-synthetic-risk", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    mapping = ActionMapping.from_yaml(config["data"]["action_mapping"])
    mapping.require_frozen()
    rows = list(read_jsonl(args.predictions))
    if not rows:
        raise RuntimeError("Prediction file is empty")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Prediction file contains duplicate sample IDs")
    synthetic = any(row.get("risk_is_synthetic") is True for row in rows)
    if synthetic and not args.allow_synthetic_risk:
        raise RuntimeError(
            "Synthetic risk predictions are infrastructure checks, not benchmark "
            "evidence; pass --allow-synthetic-risk to record that limited scope"
        )

    y_true: list[str] = []
    y_pred: list[str | None] = []
    for row in rows:
        target = row.get("target_action")
        prediction = row.get("prediction")
        valid = row.get("valid")
        if target is None:
            raise RuntimeError(f"Sample {row['id']} has no evaluable target")
        if valid is True and prediction is None:
            raise RuntimeError(f"Sample {row['id']} is marked valid without a prediction")
        if valid is False and prediction is not None:
            raise RuntimeError(f"Sample {row['id']} is invalid but has a prediction")
        y_true.append(str(target))
        y_pred.append(None if prediction is None else str(prediction))

    metrics: dict[str, Any] = {
        "classification": evaluate_actions(
            y_true,
            y_pred,
            severity=mapping.severity,
            critical_actions=config["evaluation"]["critical_actions"],
        ),
        "evidence_scope": "SYNTHETIC_INFRASTRUCTURE" if synthetic else "BENCHMARK",
    }

    timing_pairs = [
        (
            row.get("frames", {}).get("event_time"),
            row.get("frames", {}).get("prediction_cutoff"),
        )
        for row in rows
    ]
    verified_timing = [
        (float(event), float(prediction))
        for event, prediction in timing_pairs
        if event is not None and prediction is not None
    ]
    if verified_timing:
        metrics["timing"] = {
            "eligible_sample_count": len(verified_timing),
            **action_lead_time_metrics(
                [item[0] for item in verified_timing],
                [item[1] for item in verified_timing],
                thresholds_seconds=config["evaluation"][
                    "lead_time_thresholds_seconds"
                ],
            ),
        }
    else:
        metrics["timing"] = {
            "status": "UNSUPPORTED",
            "reason": "No verified event and prediction timestamp pairs",
        }

    latency_measurements = [
        LatencyMeasurement(
            latency_seconds=float(row["latency_seconds"]),
            peak_vram_bytes=row.get("peak_vram_bytes"),
            device=str(row.get("model", {}).get("device", "unknown")),
        )
        for row in rows
        if row.get("latency_seconds") is not None
    ]
    latency = summarize_latency(latency_measurements) if latency_measurements else None
    split_versions = {row.get("split_version") for row in rows}
    if len(split_versions) != 1:
        raise RuntimeError(f"Mixed split versions in predictions: {split_versions}")
    config["data"]["split_version"] = next(iter(split_versions))
    config["data"]["action_mapping_version"] = mapping.version

    destination = write_experiment(
        config=config,
        metrics=metrics,
        status="MEASURED",
        output_root=args.output_root,
        experiment_id=args.experiment_id,
        latency=latency,
        peak_vram_bytes=max(
            (
                int(row["peak_vram_bytes"])
                for row in rows
                if row.get("peak_vram_bytes") is not None
            ),
            default=None,
        ),
        notes=(
            ["Synthetic risk was explicitly allowed; not benchmark evidence."]
            if synthetic
            else []
        ),
    )
    print(destination)


if __name__ == "__main__":
    main()
