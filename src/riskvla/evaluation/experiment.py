"""Machine-readable experiment governance and provenance."""

from __future__ import annotations

import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _git(command: list[str], repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_provenance(repository: str | Path = ".") -> dict[str, Any]:
    repo = Path(repository).resolve()
    commit = _git(["rev-parse", "HEAD"], repo)
    status = _git(["status", "--porcelain"], repo)
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def hardware_provenance() -> dict[str, Any]:
    record: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": None,
        "cuda": None,
        "torch": None,
    }
    try:
        import torch
    except ImportError:
        return record
    record["torch"] = torch.__version__
    record["cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(0)
    return record


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_experiment(
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    status: str,
    output_root: str | Path = "outputs/experiments",
    repository: str | Path = ".",
    experiment_id: str | None = None,
    latency: dict[str, Any] | None = None,
    peak_vram_bytes: int | None = None,
    notes: list[str] | None = None,
) -> Path:
    if status not in {"MEASURED", "BLOCKED", "FAILED"}:
        raise ValueError("status must be MEASURED, BLOCKED, or FAILED")
    timestamp = datetime.now(timezone.utc)
    if experiment_id is None:
        name = str(config.get("experiment", {}).get("name", "experiment"))
        safe_name = "".join(character if character.isalnum() else "-" for character in name)
        safe_name = safe_name.strip("-").lower() or "experiment"
        experiment_id = (
            f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{safe_name}-"
            f"{uuid.uuid4().hex[:8]}"
        )
    destination = Path(output_root).resolve() / experiment_id
    destination.mkdir(parents=True, exist_ok=False)

    data_config = config.get("data", {})
    vla_config = config.get("vla", {})
    risk_config = config.get("risk", {})
    record = {
        "experiment_id": experiment_id,
        "timestamp": timestamp.isoformat(),
        "status": status,
        **git_provenance(repository),
        "config_file": "config.yaml",
        "dataset_split_version": data_config.get("split_version"),
        "action_mapping_version": data_config.get("action_mapping_version"),
        "model_version": {
            "id": vla_config.get("model_id"),
            "revision": vla_config.get("model_revision"),
        },
        "risk_provider_version": {
            "id": risk_config.get("model_id") or risk_config.get("provider"),
            "revision": risk_config.get("model_revision"),
        },
        "seed": config.get("project", {}).get("seed"),
        "precision": vla_config.get("precision"),
        "hardware": hardware_provenance(),
        "metrics": metrics,
        "latency": latency,
        "peak_vram_bytes": peak_vram_bytes,
        "notes": notes or [],
    }
    (destination / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (destination / "metrics.json").write_text(
        json.dumps(record, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return destination
