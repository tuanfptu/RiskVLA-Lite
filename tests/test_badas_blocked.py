from __future__ import annotations

from pathlib import Path

import pytest

from riskvla.risk.badas import BadasBlockedError, BADASRiskProvider


def test_missing_hf_access_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    provider = BADASRiskProvider(device="cpu")
    status = provider.runtime_status()
    assert status["status"] == "BLOCKED"
    assert status["metrics"] is None
    with pytest.raises(BadasBlockedError, match="BLOCKED") as caught:
        provider.load()
    assert caught.value.status == "BLOCKED"
    assert "hf_access_unavailable" in {item["code"] for item in caught.value.blockers}


def test_missing_gpu_and_checkpoint_are_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("riskvla.risk.badas.cuda_available", lambda: False)
    provider = BADASRiskProvider(
        checkpoint_path=tmp_path / "missing.pth",
        device="cuda",
    )
    with pytest.raises(BadasBlockedError, match="GPU is unavailable") as caught:
        provider.predict(tmp_path / "video.mp4")
    codes = {item["code"] for item in caught.value.blockers}
    assert {"gpu_unavailable", "checkpoint_cannot_be_loaded"} <= codes


def test_unreadable_checkpoint_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "badas_open.pth"
    checkpoint.write_bytes(b"not-a-checkpoint")
    monkeypatch.setattr("riskvla.risk.badas.badas_source_available", lambda: False)
    provider = BADASRiskProvider(checkpoint_path=checkpoint, device="cpu")
    with pytest.raises(BadasBlockedError, match="Checkpoint cannot be loaded") as caught:
        provider.load()
    assert all(
        item["code"] == "checkpoint_cannot_be_loaded" for item in caught.value.blockers
    )
