"""Lazy Qwen3-VL action selector with deterministic decoding and strict parsing."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from riskvla.risk.features import RiskFeatures
from riskvla.vision.preprocessing import prepare_frames
from riskvla.vla.action_parser import (
    ActionPrediction,
    InvalidActionOutput,
    parse_action_output,
)
from riskvla.vla.conditioning import ConditioningVariant
from riskvla.vla.prompting import build_action_prompt


class QwenUnavailableError(RuntimeError):
    """Raised when the supported Qwen runtime cannot be loaded."""


@dataclass(frozen=True)
class QwenConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    revision: str = "main"
    device: str = "cuda"
    precision: str = "bf16"
    attention_implementation: str = "sdpa"
    max_new_tokens: int = 64
    max_pixels_per_frame: int | None = 50_176

    def __post_init__(self) -> None:
        if self.precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("precision must be bf16, fp16, or fp32")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True)
class QwenPredictionResult:
    prediction: ActionPrediction | None
    raw_output: str
    valid: bool
    invalid_reason: str | None
    latency_seconds: float
    peak_vram_bytes: int | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prediction"] = (
            self.prediction.to_dict() if self.prediction is not None else None
        )
        return payload


class QwenActionSelector:
    """Replaceable action selector backed by official Transformers classes."""

    def __init__(self, config: QwenConfig | None = None) -> None:
        self.config = config or QwenConfig()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._actual_device = ""

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import transformers
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise QwenUnavailableError(
                "Install the inference extra to run Qwen3-VL: "
                'python -m pip install -e ".[inference]"'
            ) from exc

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise QwenUnavailableError(
                "Qwen configuration requests CUDA, but Torch reports no CUDA device"
            )
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.config.precision]
        if (
            self.config.precision == "bf16"
            and self.config.device.startswith("cuda")
            and not torch.cuda.is_bf16_supported()
        ):
            raise QwenUnavailableError(
                "BF16 was requested but the installed CUDA/PyTorch stack does not "
                "report BF16 support; select fp16 explicitly"
            )

        device_map: str | None = "auto" if self.config.device.startswith("cuda") else None
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                dtype=dtype,
                device_map=device_map,
                attn_implementation=self.config.attention_implementation,
            ).eval()
            if device_map is None:
                model = model.to(self.config.device)
            processor = AutoProcessor.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
            )
        except Exception as exc:
            raise QwenUnavailableError(
                f"Failed to load official model {self.config.model_id!r} at "
                f"revision {self.config.revision!r}"
            ) from exc

        self._torch = torch
        self._model = model
        self._processor = processor
        self._actual_device = str(next(model.parameters()).device)
        self._transformers_version = transformers.__version__

    def predict(
        self,
        frames: Iterable[str | Path | Image.Image],
        *,
        variant: ConditioningVariant | str,
        risk_features: RiskFeatures | None = None,
        diagnostic: bool = False,
    ) -> QwenPredictionResult:
        self.load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        prepared = prepare_frames(
            frames, max_pixels=self.config.max_pixels_per_frame
        )
        prompt = build_action_prompt(
            variant=variant,
            risk_features=risk_features,
            diagnostic=diagnostic,
        )
        content: list[dict[str, Any]] = [
            {"type": "image", "image": frame} for frame in prepared
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._actual_device)
        except Exception as exc:
            raise QwenUnavailableError("Qwen processor rejected the frame prompt") from exc

        is_cuda = self._actual_device.startswith("cuda")
        if is_cuda:
            self._torch.cuda.synchronize()
            self._torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.config.max_new_tokens,
            )
        if is_cuda:
            self._torch.cuda.synchronize()
        latency = time.perf_counter() - started
        peak_vram = (
            int(self._torch.cuda.max_memory_allocated()) if is_cuda else None
        )

        prompt_length = int(inputs["input_ids"].shape[1])
        generated_only = generated[:, prompt_length:]
        raw_output = self._processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        try:
            prediction = parse_action_output(
                raw_output, allow_diagnostics=diagnostic
            )
            valid = True
            invalid_reason = None
        except InvalidActionOutput as exc:
            prediction = None
            valid = False
            invalid_reason = str(exc)

        return QwenPredictionResult(
            prediction=prediction,
            raw_output=raw_output,
            valid=valid,
            invalid_reason=invalid_reason,
            latency_seconds=latency,
            peak_vram_bytes=peak_vram,
            metadata={
                "model_id": self.config.model_id,
                "model_revision": self.config.revision,
                "transformers_version": self._transformers_version,
                "device": self._actual_device,
                "precision": self.config.precision,
                "attention_implementation": self.config.attention_implementation,
                "frame_count": len(prepared),
                "max_new_tokens": self.config.max_new_tokens,
                "decoding": "greedy",
            },
        )
