from __future__ import annotations

import pytest

from riskvla.vla.action_parser import InvalidActionOutput, parse_action_output


def test_valid_primary_action_output() -> None:
    result = parse_action_output('{"action":"SLOW","confidence":0.87}')
    assert result.action == "SLOW"
    assert result.confidence == pytest.approx(0.87)


def test_diagnostic_output_requires_explicit_mode() -> None:
    raw = (
        '{"hazard_type":"pedestrian","ego_relation":"ahead",'
        '"action":"BRAKE_OR_STOP","confidence":0.9}'
    )
    with pytest.raises(InvalidActionOutput, match="Unexpected keys"):
        parse_action_output(raw)
    result = parse_action_output(raw, allow_diagnostics=True)
    assert result.hazard_type == "pedestrian"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("SLOW", "valid JSON"),
        ('{"action":"slow","confidence":0.5}', "Unknown action"),
        ('{"action":"SLOW"}', "Missing required"),
        ('{"action":"SLOW","confidence":1.1}', "within"),
        ('{"action":"SLOW","confidence":true}', "must be a number"),
        (
            'Here: {"action":"SLOW","confidence":0.5}',
            "valid JSON",
        ),
        (
            '{"action":"SLOW","confidence":0.5,"reason":"because"}',
            "Unexpected keys",
        ),
    ],
)
def test_invalid_output_is_never_silently_mapped(raw: str, message: str) -> None:
    with pytest.raises(InvalidActionOutput, match=message):
        parse_action_output(raw)
