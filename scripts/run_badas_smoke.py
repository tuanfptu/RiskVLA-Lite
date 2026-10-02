#!/usr/bin/env python3
"""Run the official BADAS model on one real video and record evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from riskvla.risk.badas import (
    EXPECTED_CHECKPOINT_BYTES,
    BadasBlockedError,
    BADASRiskProvider,
)


def probe_video(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install the inference extra for OpenCV video probing") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open {path}")
        return {
            "source_fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "source_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/badas_smoke.json"))
    parser.add_argument("--allow-load-warnings", action="store_true")
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    provider = BADASRiskProvider(
        checkpoint_path=args.checkpoint,
        device=args.device,
        strict_load_warnings=not args.allow_load_warnings,
    )
    try:
        video_metadata = probe_video(video)
        sequence = provider.predict(video)
    except BadasBlockedError as exc:
        blocked = {
            "status": exc.status,
            "passed": False,
            "blockers": exc.blockers,
            "metrics": None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(blocked, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(exc) from exc
    scores = sequence.risk_scores
    warmup = min(16, scores.size)
    checks = {
        "checkpoint_size_matches_official": (
            sequence.metadata["checkpoint_bytes"] == EXPECTED_CHECKPOINT_BYTES
        ),
        "output_is_vector": scores.ndim == 1 and scores.size > 16,
        "first_16_are_nan": bool(np.isnan(scores[:warmup]).all()),
        "post_warmup_all_finite": bool(np.isfinite(scores[16:]).all()),
        "post_warmup_in_unit_interval": bool(
            ((scores[16:] >= 0.0) & (scores[16:] <= 1.0)).all()
        ),
        "timeline_matches_scores": sequence.timestamps.shape == scores.shape,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"BADAS smoke checks failed: {failed}")

    report = {
        "status": "MEASURED",
        "passed": True,
        "input": {"path": str(video), **video_metadata},
        "preprocessing_contract": {
            "target_fps": 8.0,
            "context_frames": 16,
            "effective_model_input": [1, 16, 3, 256, 256],
        },
        "output": {
            "score_count": int(scores.size),
            "finite_count": sequence.valid_count,
            "warmup_nan_count": int(np.isnan(scores).sum()),
            "minimum_finite_score": float(np.nanmin(scores)),
            "maximum_finite_score": float(np.nanmax(scores)),
            "semantics": sequence.metadata["score_semantics"],
        },
        "runtime": sequence.metadata,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
