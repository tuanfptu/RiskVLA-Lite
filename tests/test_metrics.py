from __future__ import annotations

import pytest

from riskvla.evaluation.metrics import evaluate_actions
from riskvla.evaluation.timing import action_lead_time_metrics, actionable_time_error


def test_safety_metrics_distinguish_under_and_over_reaction() -> None:
    result = evaluate_actions(
        ["BRAKE_OR_STOP", "MAINTAIN", "SLOW", "CAUTION"],
        ["CAUTION", "BRAKE_OR_STOP", "SLOW", None],
    )
    assert result["accuracy"] == pytest.approx(0.25)
    assert result["critical_action_recall"] == 0.0
    assert result["under_reaction"]["count"] == 2
    assert result["over_reaction"]["count"] == 1
    assert result["unnecessary_braking"]["count"] == 1
    assert result["invalid_output_rate"] == pytest.approx(0.25)
    assert result["labels"] == ["MAINTAIN", "CAUTION", "SLOW", "BRAKE_OR_STOP"]


def test_invalid_critical_output_counts_as_missed_and_under_reaction() -> None:
    result = evaluate_actions(["BRAKE_OR_STOP"], [None])
    assert result["critical_action_recall"] == 0.0
    assert result["under_reaction"]["rate"] == 1.0
    assert result["confusion_matrix"][3][-1] == 1


def test_lead_time_definition_and_thresholds() -> None:
    result = action_lead_time_metrics(
        event_times=[5.0, 3.0, 4.0],
        prediction_times=[3.0, 2.5, 4.5],
    )
    assert result["lead_times_seconds"] == [2.0, 0.5, -0.5]
    assert result["positive_lead_rate"] == pytest.approx(2 / 3)
    assert result["threshold_rates"]["at_least_1_seconds"] == pytest.approx(1 / 3)


def test_lead_time_requires_real_pairs() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        action_lead_time_metrics([], [])


def test_event_lead_time_and_actionable_time_error_are_distinct() -> None:
    lead = action_lead_time_metrics(
        event_times=[10.0],
        prediction_times=[8.0],
    )
    error = actionable_time_error(
        predicted_action_times=[8.0],
        human_actionable_times=[7.0],
    )
    assert lead["lead_times_seconds"] == [2.0]
    assert error["errors_seconds"] == [1.0]
    assert lead["definition"] != error["definition"]
    assert "human_actionable_time" in lead["distinct_from"]
    assert "time_of_event" in error["distinct_from"]
