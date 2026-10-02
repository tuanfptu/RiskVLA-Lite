"""Streamlit dashboard separating risk timing from action selection."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from riskvla.risk.badas import BADASRiskProvider
from riskvla.risk.features import TriggerConfig, extract_risk_features
from riskvla.risk.mock import MockRiskProvider
from riskvla.vision.frame_sampler import (
    UniformFrameSampler,
    decode_selected_frames,
    probe_video_timestamps,
)
from riskvla.vla.conditioning import ConditioningVariant
from riskvla.vla.qwen import QwenActionSelector, QwenConfig


def _synthetic_action(current: float, slope: float) -> tuple[str, float]:
    """UI plumbing only; deliberately not presented as learned VLA output."""
    if current > 0.8:
        return "BRAKE_OR_STOP", min(0.99, 0.55 + current / 2)
    if slope > 0.2 or current > 0.65:
        return "SLOW", min(0.95, 0.5 + current / 2)
    if current > 0.4:
        return "CAUTION", min(0.9, 0.5 + current / 3)
    return "MAINTAIN", max(0.5, 1.0 - current)


def main() -> None:
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError('Install the demo extra: pip install -e ".[demo]"') from exc

    st.set_page_config(page_title="RiskVLA-Lite", layout="wide")
    st.title("RiskVLA-Lite")
    st.warning(
        "Research prototype only. Not a certified driver-assistance or "
        "vehicle-control system."
    )
    st.caption(
        "Risk detector: WHEN is danger emerging?  •  "
        "VLA: WHAT should the ego vehicle do?"
    )

    uploaded = st.file_uploader("Upload a dashcam video", type=["mp4", "avi", "mov", "mkv"])
    mode = st.radio(
        "Execution mode",
        [
            "Synthetic plumbing preview (not model evidence)",
            "Live official BADAS + Qwen (requires access and GPU)",
        ],
    )
    known_event = st.checkbox("A verified event timestamp is available")
    event_time = (
        st.number_input("Verified event time (seconds)", min_value=0.01, value=3.0)
        if known_event
        else None
    )
    run = st.button("Run", type="primary", disabled=uploaded is None)
    if uploaded is None:
        st.info("Upload a video to inspect the research interface.")
        return

    video_bytes = uploaded.getvalue()
    left, right = st.columns([3, 2])
    with left:
        st.video(video_bytes)
    if not run:
        return

    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(video_bytes)
        video_path = Path(handle.name)
    try:
        timeline = probe_video_timestamps(video_path)
        cutoff = float(timeline[-1])
        if event_time is not None:
            cutoff = min(cutoff, float(np.nextafter(event_time, -np.inf)))
        selection = UniformFrameSampler(4).select(
            timeline,
            prediction_cutoff=cutoff,
            event_time=event_time,
        )
        frames = decode_selected_frames(video_path, selection)

        if mode.startswith("Synthetic"):
            risk_timestamps = np.arange(0.0, cutoff + 1e-9, 0.125)
            if risk_timestamps.size < 2:
                risk_timestamps = np.array([0.0, max(cutoff, 0.125)])
            midpoint = 0.65 * max(float(risk_timestamps[-1]), 0.125)
            risk_scores = 1.0 / (
                1.0 + np.exp(-3.0 * (risk_timestamps - midpoint))
            )
            provider = MockRiskProvider(
                timestamps=risk_timestamps,
                risk_scores=risk_scores,
                require_media=True,
            )
            sequence = provider.predict(video_path)
            features = extract_risk_features(
                sequence,
                prediction_timestamp=cutoff,
                trigger=TriggerConfig(),
            )
            action, confidence = _synthetic_action(
                features.current_risk, features.risk_slope
            )
            action_source = "SYNTHETIC RULE — NOT QWEN"
            latency: dict[str, Any] = {}
        else:
            with st.spinner("Running official frozen BADAS and Qwen..."):
                provider = BADASRiskProvider(device="cuda")
                sequence = provider.predict(video_path)
                risk_cutoff = min(cutoff, sequence.duration_seconds)
                features = extract_risk_features(
                    sequence,
                    prediction_timestamp=risk_cutoff,
                    trigger=TriggerConfig(),
                )
                selector = QwenActionSelector(QwenConfig())
                result = selector.predict(
                    frames,
                    variant=ConditioningVariant.VISUAL_PLUS_TEMPORAL_RISK,
                    risk_features=features,
                )
            if not result.valid or result.prediction is None:
                st.error(f"Qwen returned invalid structured output: {result.invalid_reason}")
                return
            action = result.prediction.action
            confidence = result.prediction.confidence
            action_source = "QWEN3-VL-2B-INSTRUCT"
            latency = {
                "badas_seconds": sequence.metadata.get("latency_seconds"),
                "vla_seconds": result.latency_seconds,
                "peak_vram_bytes": result.peak_vram_bytes,
            }

        with right:
            st.subheader("Frozen risk detector")
            first, second, third = st.columns(3)
            first.metric("Risk score", f"{features.current_risk:.3f}")
            second.metric("Risk trend / s", f"{features.risk_slope:+.3f}")
            third.metric("Trigger", "TRIGGERED" if features.triggered else "IDLE")
            st.subheader("RiskVLA action")
            st.metric("Recommended action", action)
            st.metric("Confidence", f"{confidence:.2f}")
            st.caption(action_source)
            if latency:
                st.json(latency)

        figure = go.Figure()
        valid = sequence.valid_mask
        figure.add_trace(
            go.Scatter(
                x=sequence.timestamps[valid],
                y=sequence.risk_scores[valid],
                mode="lines",
                name="risk score",
            )
        )
        if features.trigger_timestamp is not None:
            figure.add_vline(
                x=features.trigger_timestamp,
                line_dash="dash",
                line_color="orange",
                annotation_text="VLA trigger",
            )
        figure.add_vline(
            x=features.prediction_timestamp,
            line_color="green",
            annotation_text="action",
        )
        if event_time is not None:
            figure.add_vline(
                x=event_time,
                line_color="red",
                annotation_text="verified event",
            )
            st.metric(
                "Action lead time",
                f"{event_time - features.prediction_timestamp:.2f} s",
            )
        figure.update_layout(
            title="Risk timeline",
            xaxis_title="time (seconds)",
            yaxis_title="risk score",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(figure, use_container_width=True)
        if mode.startswith("Synthetic"):
            st.error(
                "SYNTHETIC PREVIEW: risk and action are generated test fixtures. "
                "They are not BADAS/Qwen outputs and must not be reported as results."
            )
    except Exception as exc:
        st.exception(exc)
    finally:
        video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
