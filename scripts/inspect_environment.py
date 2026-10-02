#!/usr/bin/env python3
"""Record software and accelerator facts without assuming a GPU exists."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_environment() -> dict[str, Any]:
    packages = [
        "riskvla-lite",
        "numpy",
        "Pillow",
        "PyYAML",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "peft",
        "huggingface-hub",
        "opencv-python-headless",
        "albumentations",
        "streamlit",
    ]
    result: dict[str, Any] = {
        "status": "MEASURED",
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {name: package_version(name) for name in packages},
        "torch": {
            "installed": False,
            "cuda_available": False,
            "cuda_version": None,
            "device_count": 0,
            "devices": [],
            "bf16_supported": None,
        },
    }
    try:
        import torch
    except ImportError:
        return result

    torch_record = result["torch"]
    torch_record["installed"] = True
    torch_record["cuda_available"] = torch.cuda.is_available()
    torch_record["cuda_version"] = torch.version.cuda
    torch_record["device_count"] = torch.cuda.device_count()
    torch_record["devices"] = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
        }
        for index in range(torch.cuda.device_count())
    ]
    if torch.cuda.is_available():
        torch_record["bf16_supported"] = torch.cuda.is_bf16_supported()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_environment()
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
