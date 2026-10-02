"""Plan and run an official Nexar download without logging credentials."""

from __future__ import annotations

import os
from pathlib import Path

from riskvla.data.nexar import (
    CARD_TOTAL_FILE_SIZE,
    DATASET_ID,
    DATASET_REVISION,
    USED_STORAGE_BYTES,
    VERIFIED_VIDEO_COUNTS,
)

SPLIT_ALLOW_PATTERNS: dict[str, list[str]] = {
    "train": ["README.md", "LICENSE", "train/**"],
    "test-public": [
        "README.md",
        "LICENSE",
        "test-public/**",
        "time_to_accident_test_map.csv",
    ],
    "test-private": [
        "README.md",
        "LICENSE",
        "test-private/**",
        "time_to_accident_test_map.csv",
    ],
    "test": [
        "README.md",
        "LICENSE",
        "test-public/**",
        "test-private/**",
        "time_to_accident_test_map.csv",
    ],
    "all": [
        "README.md",
        "LICENSE",
        "train/**",
        "test-public/**",
        "test-private/**",
        "time_to_accident_test_map.csv",
        "evaluate_submission.py",
        "sample_submission.csv",
        "solution.csv",
    ],
}


class NexarDownloadError(RuntimeError):
    """Raised when a download cannot start or does not finish."""


def read_hf_token() -> str | None:
    """Return a Hugging Face token from the environment, never from argv."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def sanitize_secret(message: str, secret: str | None) -> str:
    if secret:
        return message.replace(secret, "[REDACTED]")
    return message


def disk_estimate_lines(split: str) -> list[str]:
    if split not in SPLIT_ALLOW_PATTERNS:
        raise NexarDownloadError(f"Unknown split {split!r}")
    counts = ", ".join(f"{key}={value}" for key, value in VERIFIED_VIDEO_COUNTS.items())
    gigabytes = USED_STORAGE_BYTES / 1_000_000_000
    return [
        f"Dataset: {DATASET_ID}",
        f"Pinned revision: {DATASET_REVISION}",
        (
            "Published repository storage (Hugging Face usedStorage): "
            f"{USED_STORAGE_BYTES} bytes ({gigabytes:.3f} GB)."
        ),
        f"Dataset card total file size: {CARD_TOTAL_FILE_SIZE}.",
        (
            "Per-file byte sizes are not exposed by the public file listing "
            "for this gated repository, so a split-level byte estimate is not claimed."
        ),
        f"Verified video counts at this revision: {counts}.",
        f"Requested split: {split}.",
        "Allow patterns: " + ", ".join(SPLIT_ALLOW_PATTERNS[split]),
        "Re-running the download resumes incomplete files in the output directory.",
        "Video files are not committed by this repository.",
    ]


def download_nexar(
    output: str | Path,
    *,
    split: str = "all",
    revision: str = DATASET_REVISION,
    token: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Download the pinned official dataset into ``output``.

    A dry run prints the disk estimate and does not contact Hugging Face.
    """
    destination = Path(output).expanduser()
    lines = disk_estimate_lines(split)
    print("\n".join(lines), flush=True)
    if dry_run:
        print("Dry run: no files will be downloaded.", flush=True)
        return destination
    if not token:
        raise NexarDownloadError(
            "HF_TOKEN is not set. The Nexar dataset is gated, so no download was started."
        )
    destination.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise NexarDownloadError(
            "huggingface-hub is not installed. No download was started."
        ) from exc
    try:
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=revision,
            local_dir=str(destination),
            allow_patterns=SPLIT_ALLOW_PATTERNS[split],
            token=token,
        )
    except Exception as exc:
        raise NexarDownloadError(
            sanitize_secret(f"Nexar download failed: {exc}", token)
        ) from None
    return destination
