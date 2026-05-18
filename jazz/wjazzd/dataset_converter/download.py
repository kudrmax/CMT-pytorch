"""Idempotent download of wjazzd.db from jazzomat.hfm-weimar.de.

Source: https://jazzomat.hfm-weimar.de/downloads/wjazzd.db
License: ODbL (Open Data Commons Open Database License)
Version: r2.2/v2.3 (2018-11-09), 456 solo transcriptions.

The file is committed to git for reproducibility (committed under
jazz/wjazzd/data/wjazzd.db). This script is for fresh setups or
manual re-download (idempotent — skips if size already matches).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

WJAZZD_URL = "https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db"
EXPECTED_SIZE_BYTES = 42_512_384  # ~40.5 MB; verified against local ~/Downloads/wjazzd.db


def _http_download(url: str, path: Path) -> None:
    """Download URL to path. Separated for testability via mock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(path, "wb") as out:
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            out.write(chunk)


def ensure_wjazzd_db(target: Path) -> None:
    """Ensure target path contains wjazzd.db with expected size.

    Idempotent: skips download if file already exists with correct size.
    Re-downloads if file exists but corrupted (wrong size).
    """
    if target.is_file() and target.stat().st_size == EXPECTED_SIZE_BYTES:
        return
    _http_download(WJAZZD_URL, target)
    if target.stat().st_size != EXPECTED_SIZE_BYTES:
        raise RuntimeError(
            f"Downloaded file size mismatch: expected {EXPECTED_SIZE_BYTES}, "
            f"got {target.stat().st_size}"
        )


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "data" / "wjazzd.db"
    print(f"Ensuring {target} ...")
    ensure_wjazzd_db(target)
    print(f"OK: {target.stat().st_size} bytes")
