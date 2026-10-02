"""Guarded language-attention LoRA setup for Qwen3-VL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LoraSettings:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("At least one LoRA target module is required")


def inspect_lora_targets(model: Any, settings: LoraSettings) -> dict[str, list[str]]:
    matches = {target: [] for target in settings.target_modules}
    for name, _module in model.named_modules():
        for target in settings.target_modules:
            if name.endswith(f".{target}") and ".language_model.layers." in name:
                matches[target].append(name)
    missing = [target for target, names in matches.items() if not names]
    if missing:
        raise RuntimeError(
            "Actual model modules do not match the declared Qwen language-attention "
            f"targets: missing {missing}"
        )
    accidental_vision = [
        name for names in matches.values() for name in names if ".visual." in name
    ]
    if accidental_vision:
        raise RuntimeError(f"LoRA target inspection matched visual modules: {accidental_vision}")
    return matches


def prepare_language_lora(
    model: Any,
    *,
    settings: LoraSettings | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Attach LoRA only after validating actual runtime module names."""
    selected = settings or LoraSettings()
    matches = inspect_lora_targets(model, selected)
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Install the train extra before configuring LoRA") from exc

    config = LoraConfig(
        r=selected.rank,
        lora_alpha=selected.alpha,
        lora_dropout=selected.dropout,
        target_modules=list(selected.target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    adapted = get_peft_model(model, config)
    # Defensive policy: even if PEFT matching behavior changes, visual weights stay frozen.
    for name, parameter in adapted.named_parameters():
        if ".visual." in name:
            parameter.requires_grad = False

    total = sum(parameter.numel() for parameter in adapted.parameters())
    trainable = sum(
        parameter.numel() for parameter in adapted.parameters() if parameter.requires_grad
    )
    if trainable == 0:
        raise RuntimeError("LoRA setup produced no trainable parameters")
    return adapted, {
        "settings": asdict(selected),
        "matched_modules": matches,
        "trainable_parameter_count": trainable,
        "total_parameter_count": total,
        "trainable_percentage": 100.0 * trainable / total,
        "visual_encoder_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in adapted.named_parameters()
            if ".visual." in name and parameter.requires_grad
        ),
    }
