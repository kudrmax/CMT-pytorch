"""YAML I/O for hparams files used by CMT trainer.

The orchestrator reads a baseline `hparams_jazz_{N}bars.yaml` (committed,
one per context-length config) and derives per-phase hparams files via
`derive_phase_hparams` — the baseline file is never modified.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_hparams(path: Path) -> dict[str, Any]:
    """Read YAML file, return dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_hparams(path: Path, data: dict[str, Any]) -> None:
    """Write dict to YAML file. Creates parent dirs if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def derive_phase_hparams(
    src: Path,
    dst: Path,
    *,
    asset_root: Path,
    data_path: Path,
    restore_rhythm_epoch: int | None = None,
    max_epoch_override: int | None = None,
) -> None:
    """Read src, override paths and optionally restore_rhythm.epoch, write to dst.

    Used to derive Phase 1 / Phase 2 hparams files from baseline `hparams_jazz.yaml`.
    Authorial run.py reads asset_root and data_io.path from hparams to determine
    where to write results and where to read pkl files.

    Args:
        src: Source hparams YAML (baseline, never modified).
        dst: Destination path (must differ from src, parent dirs created).
        asset_root: Overrides hparams["asset_root"] — where run.py writes results.
        data_path: Overrides hparams["data_io"]["path"] — where run.py reads pkl files.
        restore_rhythm_epoch: If given, sets experiment.restore_rhythm.epoch (Phase 2).
        max_epoch_override: If given, overrides experiment.max_epoch (e.g. smoke tests).
    """
    if src == dst:
        raise ValueError("src and dst must be different paths")
    data = load_hparams(src)
    data["asset_root"] = str(asset_root)
    data.setdefault("data_io", {})["path"] = str(data_path)
    if restore_rhythm_epoch is not None:
        data.setdefault("experiment", {}).setdefault("restore_rhythm", {})["epoch"] = int(restore_rhythm_epoch)
    if max_epoch_override is not None:
        data.setdefault("experiment", {})["max_epoch"] = int(max_epoch_override)
    save_hparams(dst, data)


