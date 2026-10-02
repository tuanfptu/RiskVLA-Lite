"""Shared zero-shot inference implementation for the A/B/C scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from riskvla.config import load_config, validate_experiment_config
from riskvla.io import read_json, read_jsonl, write_jsonl
from riskvla.risk.features import RiskFeatures
from riskvla.vision.frame_sampler import (
    UniformFrameSampler,
    decode_selected_frames,
    probe_video_timestamps,
)
from riskvla.vla.conditioning import ConditioningVariant
from riskvla.vla.qwen import QwenActionSelector, QwenConfig


def _resolve_media(raw_path: str, media_root: Path | None) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if media_root is not None:
        rooted = (media_root / candidate).resolve()
        if rooted.is_file():
            return rooted
    raise FileNotFoundError(
        f"Media path is not a local file: {raw_path!r}. Download licensed Nexar "
        "media separately and pass --media-root if paths are relative."
    )


def _frames_for_sample(
    sample: dict[str, Any],
    *,
    frame_count: int,
    media_root: Path | None,
) -> tuple[list[Any], dict[str, Any]]:
    video_path = str(sample.get("video_path", "")).strip()
    image_path = str(sample.get("image_path", "")).strip()
    if video_path:
        resolved = _resolve_media(video_path, media_root)
        timeline = probe_video_timestamps(resolved)
        cutoff_value = sample.get("observation_cutoff", sample.get("prediction_time"))
        if cutoff_value is None:
            raise RuntimeError(
                f"Sample {sample.get('id')} lacks observation_cutoff; "
                "refusing to sample the full video"
            )
        event_time = sample.get("time_of_event", sample.get("event_time"))
        cutoff = float(cutoff_value)
        selection = UniformFrameSampler(frame_count).select(
            timeline,
            prediction_cutoff=cutoff,
            event_time=None if event_time is None else float(event_time),
        )
        return decode_selected_frames(resolved, selection), {
            "media_type": "video",
            "media_path": str(resolved),
            "frame_indices": list(selection.indices),
            "frame_timestamps": list(selection.timestamps),
            "prediction_cutoff": selection.prediction_cutoff,
            "event_time": selection.event_time,
        }
    if image_path:
        resolved = _resolve_media(image_path, media_root)
        return [resolved], {
            "media_type": "image",
            "media_path": str(resolved),
            "frame_indices": None,
            "frame_timestamps": None,
            "prediction_cutoff": None,
            "event_time": None,
        }
    raise RuntimeError(f"Sample {sample.get('id')} has no populated media reference")


def run_zero_shot(
    *,
    config_path: Path,
    manifest_path: Path,
    output_path: Path,
    variant: ConditioningVariant,
    risk_features_path: Path | None = None,
    media_root: Path | None = None,
    limit: int | None = None,
    allow_synthetic_risk: bool = False,
) -> None:
    config = load_config(config_path)
    validate_experiment_config(config)
    configured_variant = ConditioningVariant(config["vla"]["variant"])
    if configured_variant is not variant:
        raise ValueError(
            f"Config variant {configured_variant.value!r} does not match "
            f"requested {variant.value!r}"
        )
    manifest = read_json(manifest_path)
    if manifest.get("status") != "READY":
        raise RuntimeError(
            f"Manifest is not READY ({manifest.get('status')}); benchmark inference blocked"
        )
    samples = manifest.get("samples", [])
    if not samples:
        raise RuntimeError("Ready manifest contains no samples")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        samples = samples[:limit]

    risk_by_id: dict[str, dict[str, Any]] = {}
    if variant is not ConditioningVariant.VISUAL_ONLY:
        if risk_features_path is None:
            raise ValueError("Risk-conditioned inference requires --risk-features")
        risk_by_id = {str(item["id"]): item for item in read_jsonl(risk_features_path)}

    qwen_config = QwenConfig(
        model_id=config["vla"]["model_id"],
        revision=str(config["vla"].get("model_revision", "main")),
        device=config["vla"].get("device", "cuda"),
        precision=config["vla"].get("precision", "bf16"),
        attention_implementation=config["vla"].get(
            "attention_implementation", "sdpa"
        ),
        max_new_tokens=int(config["vla"].get("max_new_tokens", 64)),
        max_pixels_per_frame=config["vision"].get("max_pixels_per_frame"),
    )
    selector = QwenActionSelector(qwen_config)
    frame_count = int(config["vision"]["frame_count"])

    predictions: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["id"])
        frames, frame_metadata = _frames_for_sample(
            sample,
            frame_count=frame_count,
            media_root=media_root,
        )
        features: RiskFeatures | None = None
        risk_record: dict[str, Any] | None = None
        if variant is not ConditioningVariant.VISUAL_ONLY:
            if sample_id not in risk_by_id:
                raise RuntimeError(f"No risk features for sample {sample_id}")
            risk_record = risk_by_id[sample_id]
            if risk_record.get("synthetic") and not allow_synthetic_risk:
                raise RuntimeError(
                    "Synthetic risk features cannot be used for a benchmark run "
                    "without --allow-synthetic-risk"
                )
            features = RiskFeatures(**risk_record["features"])

        result = selector.predict(
            frames,
            variant=variant,
            risk_features=features,
        )
        predictions.append(
            {
                "id": sample_id,
                "group_id": sample.get("group_id"),
                "target_action": sample.get("action", sample.get("label")),
                "variant": variant.value,
                "prediction": (
                    result.prediction.action if result.prediction is not None else None
                ),
                "confidence": (
                    result.prediction.confidence
                    if result.prediction is not None
                    else None
                ),
                "valid": result.valid,
                "invalid_reason": result.invalid_reason,
                "raw_output": result.raw_output,
                "risk_features": (
                    features.to_dict() if features is not None else None
                ),
                "risk_is_synthetic": (
                    bool(risk_record.get("synthetic")) if risk_record else None
                ),
                "frames": frame_metadata,
                "latency_seconds": result.latency_seconds,
                "peak_vram_bytes": result.peak_vram_bytes,
                "model": result.metadata,
                "split_version": manifest.get("split_version"),
            }
        )
    write_jsonl(output_path, predictions)
    print(f"Wrote {len(predictions)} {variant.value} predictions to {output_path}")
