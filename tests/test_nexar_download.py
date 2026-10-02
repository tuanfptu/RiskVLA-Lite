from __future__ import annotations

from riskvla.data.nexar import CARD_TOTAL_FILE_SIZE, USED_STORAGE_BYTES
from riskvla.data.nexar_download import (
    SPLIT_ALLOW_PATTERNS,
    disk_estimate_lines,
    download_nexar,
    sanitize_secret,
)


def test_disk_estimate_uses_verified_storage_and_does_not_invent_split_bytes() -> None:
    lines = "\n".join(disk_estimate_lines("train"))
    assert str(USED_STORAGE_BYTES) in lines
    assert CARD_TOTAL_FILE_SIZE in lines
    assert "train/**" in lines
    assert "not claimed" in lines


def test_secret_text_is_redacted() -> None:
    token = "hf_example_token_value"
    assert token not in sanitize_secret(f"request failed for {token}", token)
    assert "[REDACTED]" in sanitize_secret(f"request failed for {token}", token)


def test_dry_run_does_not_download(tmp_path) -> None:
    destination = download_nexar(tmp_path / "nexar", split="train", dry_run=True)
    assert destination == tmp_path / "nexar"
    assert not destination.exists()


def test_missing_token_does_not_start_a_download(tmp_path) -> None:
    from riskvla.data.nexar_download import NexarDownloadError

    try:
        download_nexar(tmp_path / "nexar", split="all", token=None, dry_run=False)
    except NexarDownloadError as exc:
        assert "HF_TOKEN" in str(exc)
        assert "no download was started" in str(exc)
    else:
        raise AssertionError("download started without a token")
    assert set(SPLIT_ALLOW_PATTERNS) == {"train", "test-public", "test-private", "test", "all"}
