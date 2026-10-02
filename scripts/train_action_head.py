#!/usr/bin/env python3
"""Train the optional frozen-representation action head after A/B/C baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from riskvla.training.action_head import (
    ActionHeadTrainingConfig,
    train_action_head,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path, help="NPZ with embeddings, labels")
    parser.add_argument("--val", required=True, type=Path, help="NPZ with embeddings, labels")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--confirm-baselines-complete", action="store_true")
    args = parser.parse_args()
    if not args.confirm_baselines_complete:
        raise RuntimeError(
            "Action-head training is gated until zero-shot A/B/C validation is complete. "
            "Pass --confirm-baselines-complete only after documenting that evidence."
        )

    train = np.load(args.train)
    val = np.load(args.val)
    settings = ActionHeadTrainingConfig(device=args.device, epochs=args.epochs)
    model, report = train_action_head(
        train["embeddings"],
        train["labels"],
        val["embeddings"],
        val["labels"],
        config=settings,
    )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch disappeared after training") from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "action_head.pt")
    payload = {
        "status": "MEASURED",
        "config": asdict(settings),
        **report,
        "warning": "Head metrics are valid only for the supplied frozen embeddings/splits.",
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
