from __future__ import annotations

import json
from pathlib import Path

import pytest

from riskvla.config import ConfigError, load_config, validate_experiment_config
from riskvla.risk.features import RiskFeatures
from riskvla.vla.conditioning import ConditioningVariant
from riskvla.vla.prompting import build_action_prompt

ROOT = Path(__file__).resolve().parents[1]


def _features() -> RiskFeatures:
    return RiskFeatures(
        prediction_timestamp=1.0,
        current_risk=0.42,
        risk_slope=0.1,
        recent_peak=0.5,
        risk_mean=0.3,
        risk_volatility=0.05,
        trigger_timestamp=0.5,
        triggered=True,
        valid_observations=4,
    )


def test_three_experiment_variants_are_config_driven() -> None:
    expected = {
        "configs/visual_only.yaml": ("visual_only", []),
        "configs/risk_score.yaml": ("visual_plus_current_risk", ["current_risk"]),
        "configs/temporal_risk.yaml": (
            "visual_plus_temporal_risk",
            ["current_risk", "risk_slope", "recent_peak"],
        ),
    }
    for relative_path, (variant, features) in expected.items():
        config = load_config(ROOT / relative_path)
        validate_experiment_config(config)
        assert config["vla"]["variant"] == variant
        assert config["vla"]["do_sample"] is False
        assert config["risk"]["features"] == features
        assert config["project"]["seed"] == 42


def test_visual_prompt_contains_no_risk_values() -> None:
    prompt = build_action_prompt(variant=ConditioningVariant.VISUAL_ONLY)
    assert "R_t" not in prompt
    assert "0.420" not in prompt
    current = build_action_prompt(
        variant="visual_plus_current_risk",
        risk_features=_features(),
    )
    temporal = build_action_prompt(
        variant="visual_plus_temporal_risk",
        risk_features=_features(),
    )
    assert "R_t: 0.420" in current
    assert "slope" not in current
    assert "Recent risk slope" in temporal
    assert "Recent maximum risk: 0.500" in temporal


def test_invalid_variant_feature_contract_fails() -> None:
    config = load_config(ROOT / "configs/risk_score.yaml")
    config["risk"]["features"] = ["current_risk", "risk_slope"]
    with pytest.raises(ConfigError, match="features must be"):
        validate_experiment_config(config)


def test_blocker_manifests_contain_no_fabricated_metrics() -> None:
    badas = json.loads(
        (ROOT / "outputs/blockers/badas_runtime.json").read_text(encoding="utf-8")
    )
    drama = json.loads(
        (ROOT / "outputs/blockers/drama_media.json").read_text(encoding="utf-8")
    )
    assert badas["access"] == "UNVERIFIED"
    assert badas["runtime"] == "BLOCKED"
    assert badas["checkpoint_download"] == "NOT_ATTEMPTED"
    assert drama["access"] == "UNVERIFIED"
    assert drama["media_runtime"] == "NOT_DOWNLOADED"
    assert drama["archive_inspection"] == "NOT_ATTEMPTED"
    for payload in (badas, drama):
        assert payload["status"] == "BLOCKED"
        assert payload["metrics"] is None
        assert payload["reasons"]
        serialized = json.dumps(payload)
        assert "hf_" not in serialized
        assert "://" not in serialized
