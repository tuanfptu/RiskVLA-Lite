#!/usr/bin/env python3
"""Optional LoRA setup. Status: NOT STARTED.

Training remains gated on zero-shot A/B/C evidence and is not part of the
Nexar dataset pivot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskvla.config import load_config
from riskvla.training.lora import LoraSettings, prepare_language_lora
from riskvla.vla.qwen import QwenActionSelector, QwenConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/temporal_risk.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/lora_setup.json"))
    parser.add_argument("--confirm-baselines-complete", action="store_true")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Load model, attach LoRA, and report parameters without optimization.",
    )
    args = parser.parse_args()
    if not args.confirm_baselines_complete:
        raise RuntimeError(
            "LoRA is intentionally gated until A/B/C validation and a documented "
            "fine-tuning rationale exist."
        )
    if not args.inspect_only:
        raise RuntimeError(
            "This proof-of-concept does not launch LoRA optimization without an "
            "approved grouped training manifest and completed zero-shot evidence. "
            "Use --inspect-only to verify target modules."
        )

    config = load_config(args.config)
    selector = QwenActionSelector(
        QwenConfig(
            model_id=config["vla"]["model_id"],
            revision=str(config["vla"].get("model_revision", "main")),
            device=config["vla"]["device"],
            precision=config["vla"]["precision"],
            attention_implementation=config["vla"]["attention_implementation"],
        )
    )
    selector.load()
    assert selector._model is not None
    _, report = prepare_language_lora(selector._model, settings=LoraSettings())
    payload = {
        "status": "MEASURED_SETUP_ONLY",
        **report,
        "optimization_performed": False,
        "visual_encoder_frozen": True,
        "badas_frozen": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
