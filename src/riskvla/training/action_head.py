"""Training loop for a classifier over pre-extracted, frozen representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from riskvla.constants import ACTIONS
from riskvla.evaluation.metrics import evaluate_actions
from riskvla.reproducibility import set_reproducible_seed
from riskvla.vla.action_head import build_action_head


@dataclass(frozen=True)
class ActionHeadTrainingConfig:
    hidden_dim: int = 256
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 20
    seed: int = 42
    device: str = "cuda"


def train_action_head(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    *,
    config: ActionHeadTrainingConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train only a small head; input embeddings are always detached arrays."""
    settings = config or ActionHeadTrainingConfig()
    if train_embeddings.ndim != 2 or val_embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape [samples, features]")
    if train_embeddings.shape[1] != val_embeddings.shape[1]:
        raise ValueError("Train and validation feature dimensions differ")
    if train_embeddings.shape[0] != train_labels.shape[0]:
        raise ValueError("Train embeddings and labels differ in length")
    if val_embeddings.shape[0] != val_labels.shape[0]:
        raise ValueError("Validation embeddings and labels differ in length")
    if not np.isfinite(train_embeddings).all() or not np.isfinite(val_embeddings).all():
        raise ValueError("Embeddings must be finite")
    if np.any(train_labels < 0) or np.any(train_labels >= len(ACTIONS)):
        raise ValueError("Train labels are outside the fixed action range")
    if np.any(val_labels < 0) or np.any(val_labels >= len(ACTIONS)):
        raise ValueError("Validation labels are outside the fixed action range")

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("Install the train extra to train an action head") from exc
    set_reproducible_seed(settings.seed)
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model = build_action_head(
        train_embeddings.shape[1],
        hidden_dim=settings.hidden_dim,
        num_actions=len(ACTIONS),
        dropout=settings.dropout,
    ).to(settings.device)
    train_dataset = TensorDataset(
        torch.from_numpy(train_embeddings).float(),
        torch.from_numpy(train_labels).long(),
    )
    generator = torch.Generator().manual_seed(settings.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    validation_x = torch.from_numpy(val_embeddings).float().to(settings.device)

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_macro_f1 = -1.0
    for epoch in range(settings.epochs):
        model.train()
        losses: list[float] = []
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(settings.device))
            loss = criterion(logits, labels.to(settings.device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.inference_mode():
            predicted_ids = model(validation_x).argmax(dim=-1).cpu().numpy()
        metrics = evaluate_actions(
            [ACTIONS[int(value)] for value in val_labels],
            [ACTIONS[int(value)] for value in predicted_ids],
        )
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
        }
        history.append(epoch_record)
        if float(metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(metrics["macro_f1"])
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {
        "config": asdict(settings),
        "history": history,
        "best_validation_macro_f1": best_macro_f1,
        "input_dimension": int(train_embeddings.shape[1]),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
