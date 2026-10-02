from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from riskvla.risk.features import extract_risk_features
from riskvla.risk.mock import MockRiskProvider


def test_mock_provider_is_deterministic_and_labeled_synthetic(tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fixture")
    provider = MockRiskProvider(
        timestamps=[0.0, 0.5, 1.0],
        risk_scores=[0.2, 0.4, 0.6],
    )
    first = provider.predict(media)
    second = provider.predict(media)
    assert np.array_equal(first.risk_scores, second.risk_scores)
    assert first.metadata["synthetic"] is True
    assert first.provider == "mock"
    features = extract_risk_features(first, prediction_timestamp=0.5)
    assert features.current_risk == pytest.approx(0.4)
    assert features.recent_peak == pytest.approx(0.4)


def test_mock_provider_requires_existing_media(tmp_path: Path) -> None:
    provider = MockRiskProvider()
    with pytest.raises(FileNotFoundError):
        provider.predict(tmp_path / "missing.mp4")
