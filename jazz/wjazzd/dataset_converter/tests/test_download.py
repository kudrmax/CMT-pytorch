"""Tests for jazz.wjazzd.dataset_converter.download module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from jazz.wjazzd.dataset_converter import download


def test_skip_if_exists(tmp_path: Path) -> None:
    """If file exists with correct size, do not re-download."""
    target = tmp_path / "wjazzd.db"
    target.write_bytes(b"x" * download.EXPECTED_SIZE_BYTES)
    with patch.object(download, "_http_download") as mock_http:
        download.ensure_wjazzd_db(target)
    mock_http.assert_not_called()


def test_downloads_when_missing(tmp_path: Path) -> None:
    """If file missing, _http_download is invoked."""
    target = tmp_path / "wjazzd.db"
    fake_payload = b"y" * download.EXPECTED_SIZE_BYTES

    def fake_http(url: str, path: Path) -> None:
        path.write_bytes(fake_payload)

    with patch.object(download, "_http_download", side_effect=fake_http) as mock_http:
        download.ensure_wjazzd_db(target)
    mock_http.assert_called_once()
    assert target.read_bytes() == fake_payload


def test_redownloads_if_size_wrong(tmp_path: Path) -> None:
    """If file exists but is wrong size, re-download."""
    target = tmp_path / "wjazzd.db"
    target.write_bytes(b"corrupted")
    with patch.object(download, "_http_download") as mock_http:
        mock_http.side_effect = lambda u, p: p.write_bytes(b"x" * download.EXPECTED_SIZE_BYTES)
        download.ensure_wjazzd_db(target)
    mock_http.assert_called_once()
