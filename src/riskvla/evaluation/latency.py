"""Device-aware latency and peak-memory measurement."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class LatencyMeasurement:
    latency_seconds: float
    peak_vram_bytes: int | None
    device: str

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


def measure_callable(
    function: Callable[..., T],
    *args: Any,
    device: str = "cpu",
    **kwargs: Any,
) -> tuple[T, LatencyMeasurement]:
    torch = None
    is_cuda = device.startswith("cuda")
    if is_cuda:
        try:
            import torch as imported_torch
        except ImportError as exc:
            raise RuntimeError("Torch is required for CUDA latency measurement") from exc
        torch = imported_torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA measurement requested but CUDA is unavailable")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    result = function(*args, **kwargs)
    if is_cuda and torch is not None:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated()) if is_cuda and torch is not None else None
    return result, LatencyMeasurement(
        latency_seconds=elapsed,
        peak_vram_bytes=peak,
        device=device,
    )


def summarize_latency(
    measurements: list[LatencyMeasurement],
) -> dict[str, float | int | str | None]:
    if not measurements:
        raise ValueError("At least one latency measurement is required")
    devices = {measurement.device for measurement in measurements}
    if len(devices) != 1:
        raise ValueError("Cannot summarize measurements from different devices")
    values = sorted(measurement.latency_seconds for measurement in measurements)
    p95_index = max(0, min(len(values) - 1, int(0.95 * len(values) + 0.9999) - 1))
    peaks = [
        measurement.peak_vram_bytes
        for measurement in measurements
        if measurement.peak_vram_bytes is not None
    ]
    return {
        "runs": len(values),
        "device": next(iter(devices)),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": values[p95_index],
        "minimum_seconds": values[0],
        "maximum_seconds": values[-1],
        "peak_vram_bytes": max(peaks) if peaks else None,
    }
