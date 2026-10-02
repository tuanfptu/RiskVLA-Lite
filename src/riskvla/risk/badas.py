"""Adapter for the official, frozen BADAS-Open runtime."""

from __future__ import annotations

import contextlib
import io
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from riskvla.risk.base import RiskProvider, RiskSequence

DEFAULT_REPO_ID = "nexar-ai/BADAS-Open"
DEFAULT_MODEL_REVISION = "8fda93711e79d72401b0a4efc151b56455885cd2"
DEFAULT_SOURCE_REVISION = "01368cf5a164a576ae46eae3baf544e4ecdb2946"
DEFAULT_CHECKPOINT_FILENAME = "weights/badas_open.pth"
EXPECTED_CHECKPOINT_BYTES = 3_979_436_545


class BadasBlockedError(RuntimeError):
    """Raised when official BADAS execution cannot start."""

    status = "BLOCKED"

    def __init__(self, blockers: list[dict[str, str]]) -> None:
        if not blockers:
            raise ValueError("A blocked BADAS result requires at least one reason")
        self.blockers = blockers
        details = "\n".join(
            f"- {blocker['code']}: {blocker['message']}" for blocker in blockers
        )
        super().__init__(f"BLOCKED: BADAS runtime cannot start.\n{details}")


class BadasUnavailableError(BadasBlockedError):
    """Compatibility alias for access and installation blockers."""


class BadasCompatibilityError(BadasBlockedError):
    """Compatibility alias for checkpoint/runtime contract failures."""


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def badas_source_available() -> bool:
    try:
        import badas
    except ImportError:
        return False
    return callable(getattr(badas, "load_badas_model", None))


class BADASRiskProvider(RiskProvider):
    name = "badas-open"
    version = DEFAULT_MODEL_REVISION

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        target_fps: float = 8.0,
        repo_id: str = DEFAULT_REPO_ID,
        model_revision: str = DEFAULT_MODEL_REVISION,
        source_revision: str = DEFAULT_SOURCE_REVISION,
        checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
        strict_load_warnings: bool = True,
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
        self.device = device
        self.target_fps = float(target_fps)
        self.repo_id = repo_id
        self.model_revision = model_revision
        self.source_revision = source_revision
        self.checkpoint_filename = checkpoint_filename
        self.strict_load_warnings = strict_load_warnings
        self._model: Any | None = None
        self._resolved_checkpoint: Path | None = None
        self._load_log = ""

    def preflight_blockers(self) -> list[dict[str, str]]:
        """Return access blockers without downloading or loading a checkpoint."""
        blockers: list[dict[str, str]] = []
        if self.device.startswith("cuda") and not cuda_available():
            blockers.append(
                {
                    "code": "gpu_unavailable",
                    "message": (
                        "GPU is unavailable. BADAS was requested on CUDA, but this "
                        "process has no usable CUDA device."
                    ),
                }
            )
        if self.checkpoint_path is not None and not self.checkpoint_path.is_file():
            blockers.append(
                {
                    "code": "checkpoint_cannot_be_loaded",
                    "message": (
                        "Checkpoint cannot be loaded because "
                        f"{self.checkpoint_path} does not exist."
                    ),
                }
            )
        elif self.checkpoint_path is None and not os.environ.get("HF_TOKEN"):
            blockers.append(
                {
                    "code": "hf_access_unavailable",
                    "message": (
                        "Hugging Face access is unavailable. The official BADAS "
                        "checkpoint is gated; request access and set HF_TOKEN. "
                        "Gating will not be bypassed."
                    ),
                }
            )
        if not badas_source_available():
            blockers.append(
                {
                    "code": "checkpoint_cannot_be_loaded",
                    "message": (
                        "Checkpoint cannot be loaded because official BADAS source "
                        f"commit {self.source_revision} is not installed."
                    ),
                }
            )
        return blockers

    def runtime_status(self) -> dict[str, Any]:
        blockers = self.preflight_blockers()
        return {
            "status": "BLOCKED" if blockers else "READY_TO_ATTEMPT_LOAD",
            "provider": self.name,
            "model_revision": self.model_revision,
            "source_revision": self.source_revision,
            "device": self.device,
            "blockers": blockers,
            "metrics": None,
        }

    def _raise_if_blocked(self) -> None:
        blockers = self.preflight_blockers()
        if blockers:
            raise BadasBlockedError(blockers)

    def _blocked_load_error(self, message: str) -> BadasBlockedError:
        return BadasBlockedError(
            [
                {
                    "code": "checkpoint_cannot_be_loaded",
                    "message": f"Checkpoint cannot be loaded: {message}",
                }
            ]
        )

    def _resolve_checkpoint(self) -> Path:
        if self.checkpoint_path is not None:
            path = self.checkpoint_path.resolve()
            if not path.is_file():
                raise self._blocked_load_error(f"{path} does not exist")
            return path

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise BadasBlockedError(
                [
                    {
                        "code": "hf_access_unavailable",
                        "message": (
                            "Hugging Face access is unavailable. The official BADAS "
                            "checkpoint is gated; request access and set HF_TOKEN."
                        ),
                    }
                ]
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise self._blocked_load_error(
                "huggingface-hub is not installed"
            ) from exc
        try:
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.checkpoint_filename,
                revision=self.model_revision,
                token=token,
            )
        except Exception as exc:
            raise BadasBlockedError(
                [
                    {
                        "code": "hf_access_unavailable",
                        "message": (
                            "Hugging Face access is unavailable or was rejected while "
                            "resolving the gated official checkpoint."
                        ),
                    }
                ]
            ) from exc
        return Path(downloaded).resolve()

    def load(self) -> None:
        if self._model is not None:
            return
        self._raise_if_blocked()
        try:
            checkpoint = self._resolve_checkpoint()
            from badas import load_badas_model

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                model = load_badas_model(
                    checkpoint_path=str(checkpoint),
                    device=self.device,
                )
        except BadasBlockedError:
            raise
        except Exception as exc:
            raise self._blocked_load_error(str(exc)) from exc
        self._load_log = captured.getvalue()
        lowered = self._load_log.lower()
        if self.strict_load_warnings and (
            "missing keys" in lowered or "unexpected keys" in lowered
        ):
            raise self._blocked_load_error(
                "BADAS reported missing or unexpected checkpoint keys. "
                f"{self._load_log.strip()}"
            )
        if not hasattr(model, "predict"):
            raise self._blocked_load_error(
                "loaded BADAS object has no predict(video_path)"
            )
        self._model = model
        self._resolved_checkpoint = checkpoint

    def predict(self, video_path: str | Path) -> RiskSequence:
        self._raise_if_blocked()
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.load()
        assert self._model is not None
        assert self._resolved_checkpoint is not None

        torch = None
        peak_vram_bytes: int | None = None
        try:
            import torch as imported_torch

            torch = imported_torch
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

        started = time.perf_counter()
        raw_scores = self._model.predict(str(path))
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram_bytes = int(torch.cuda.max_memory_allocated())
        latency_seconds = time.perf_counter() - started

        scores = np.asarray(raw_scores, dtype=np.float32)
        if scores.ndim != 1 or scores.size == 0:
            raise self._blocked_load_error(
                f"BADAS returned shape {scores.shape}; expected a non-empty vector"
            )
        timestamps = np.arange(scores.size, dtype=np.float64) / self.target_fps
        finite_count = int(np.isfinite(scores).sum())
        if finite_count == 0:
            raise self._blocked_load_error(
                "BADAS produced no finite score; the video may be too short for "
                "its 16-frame context window"
            )

        return RiskSequence(
            timestamps=timestamps,
            risk_scores=scores,
            provider=self.name,
            metadata={
                "synthetic": False,
                "model_repo": self.repo_id,
                "model_revision": self.model_revision,
                "source_revision": self.source_revision,
                "checkpoint_filename": self.checkpoint_filename,
                "checkpoint_path": str(self._resolved_checkpoint),
                "checkpoint_bytes": self._resolved_checkpoint.stat().st_size,
                "expected_checkpoint_bytes": EXPECTED_CHECKPOINT_BYTES,
                "device_requested": self.device,
                "target_fps": self.target_fps,
                "context_frames": 16,
                "score_semantics": (
                    "softmax(logits / 2) class-1 accident/near-miss score; "
                    "not established as a calibrated collision probability"
                ),
                "warmup_invalid_count": int(np.isnan(scores).sum()),
                "finite_count": finite_count,
                "latency_seconds": latency_seconds,
                "peak_vram_bytes": peak_vram_bytes,
                "load_log": self._load_log,
            },
        )


BadasRiskProvider = BADASRiskProvider
