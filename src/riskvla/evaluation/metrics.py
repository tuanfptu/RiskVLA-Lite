"""Classification metrics with explicit safety-cost asymmetry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from riskvla.constants import ACTIONS, DEFAULT_SEVERITY

INVALID_LABEL = "INVALID"


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_actions(
    y_true: Sequence[str],
    y_pred: Sequence[str | None],
    *,
    severity: Mapping[str, int] | None = None,
    critical_actions: Sequence[str] = ("BRAKE_OR_STOP",),
) -> dict[str, Any]:
    """Compute fixed-label classification and safety-oriented metrics.

    Invalid outputs are represented by ``None``. They count as classification
    errors and missed critical actions rather than being silently mapped.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        raise ValueError("At least one evaluated sample is required")
    unknown_true = sorted(set(y_true) - set(ACTIONS))
    unknown_pred = sorted({value for value in y_pred if value is not None} - set(ACTIONS))
    if unknown_true:
        raise ValueError(f"Unknown ground-truth actions: {unknown_true}")
    if unknown_pred:
        raise ValueError(f"Unknown predicted actions: {unknown_pred}")

    severity_map = dict(severity or DEFAULT_SEVERITY)
    if set(severity_map) != set(ACTIONS):
        raise ValueError("Severity mapping must define every action")
    unknown_critical = sorted(set(critical_actions) - set(ACTIONS))
    if unknown_critical:
        raise ValueError(f"Unknown critical actions: {unknown_critical}")

    prediction_labels = (*ACTIONS, INVALID_LABEL)
    true_index = {label: index for index, label in enumerate(ACTIONS)}
    pred_index = {label: index for index, label in enumerate(prediction_labels)}
    matrix = np.zeros((len(ACTIONS), len(prediction_labels)), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred, strict=True):
        matrix[true_index[truth], pred_index[prediction or INVALID_LABEL]] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in ACTIONS:
        row = true_index[label]
        column = pred_index[label]
        tp = int(matrix[row, column])
        fp = int(matrix[:, column].sum() - tp)
        fn = int(matrix[row, :].sum() - tp)
        support = int(matrix[row, :].sum())
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    total = len(y_true)
    correct = sum(
        int(truth == prediction)
        for truth, prediction in zip(y_true, y_pred, strict=True)
    )
    invalid_count = sum(prediction is None for prediction in y_pred)

    critical_mask = [truth in critical_actions for truth in y_true]
    critical_total = sum(critical_mask)
    critical_correct = sum(
        truth == prediction
        for truth, prediction, is_critical in zip(
            y_true, y_pred, critical_mask, strict=True
        )
        if is_critical
    )

    minimum_severity = min(severity_map.values())
    under_eligible = 0
    under_count = 0
    over_eligible = total
    over_count = 0
    for truth, prediction in zip(y_true, y_pred, strict=True):
        truth_severity = severity_map[truth]
        if truth_severity > minimum_severity:
            under_eligible += 1
            if prediction is None or severity_map[prediction] < truth_severity:
                under_count += 1
        if prediction is not None and severity_map[prediction] > truth_severity:
            over_count += 1

    non_brake_total = sum(truth != "BRAKE_OR_STOP" for truth in y_true)
    unnecessary_brakes = sum(
        truth != "BRAKE_OR_STOP" and prediction == "BRAKE_OR_STOP"
        for truth, prediction in zip(y_true, y_pred, strict=True)
    )

    return {
        "sample_count": total,
        "labels": list(ACTIONS),
        "prediction_labels": list(prediction_labels),
        "accuracy": _safe_ratio(correct, total),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "invalid_output_count": invalid_count,
        "invalid_output_rate": _safe_ratio(invalid_count, total),
        "critical_actions": list(critical_actions),
        "critical_action_support": critical_total,
        "critical_action_recall": _safe_ratio(critical_correct, critical_total),
        "under_reaction": {
            "count": under_count,
            "eligible": under_eligible,
            "rate": _safe_ratio(under_count, under_eligible),
            "definition": (
                "prediction severity below target; invalid output counts as "
                "under-reaction for targets above minimum severity"
            ),
        },
        "over_reaction": {
            "count": over_count,
            "eligible": over_eligible,
            "rate": _safe_ratio(over_count, over_eligible),
            "definition": "prediction severity above target",
        },
        "unnecessary_braking": {
            "count": unnecessary_brakes,
            "eligible": non_brake_total,
            "rate": _safe_ratio(unnecessary_brakes, non_brake_total),
            "definition": "BRAKE_OR_STOP predicted for a non-BRAKE_OR_STOP target",
        },
        "severity": severity_map,
    }
