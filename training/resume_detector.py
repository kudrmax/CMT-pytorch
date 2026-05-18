"""State detection for resume-aware orchestrator.

Scans result/<experiment>/training_artefacts/idxNNN/model/ to determine
training progress for each phase.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)\.pth\.tar$")


@dataclass(frozen=True)
class PhaseState:
    """Status of one training phase (RD or PD).

    status: 'not_started' | 'partial' | 'complete'
    last_epoch: max checkpoint epoch present (0 if not_started).
    """

    status: str
    last_epoch: int


def _scan_checkpoints(model_dir: Path) -> list[int]:
    """Return sorted list of epochs that have a saved checkpoint."""
    if not model_dir.is_dir():
        return []
    epochs: list[int] = []
    for entry in model_dir.iterdir():
        m = _CHECKPOINT_RE.match(entry.name)
        if m and entry.is_file():
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def detect_phase_state(asset_path: Path, max_epoch: int) -> PhaseState:
    """Determine phase status from saved checkpoints in asset_path/model/.

    trainer.py iterates range(1, max_epoch) → 1..max_epoch-1
    Saves at multiples of 10. For max_epoch=101 last save is at 100.
    Phase 'complete' when last >= final_save_epoch.
    """
    epochs = _scan_checkpoints(asset_path / "model")
    if not epochs:
        return PhaseState(status="not_started", last_epoch=0)
    last = epochs[-1]
    final_save_epoch = (max_epoch - 1) // 10 * 10
    if last >= final_save_epoch:
        return PhaseState(status="complete", last_epoch=last)
    return PhaseState(status="partial", last_epoch=last)


@dataclass(frozen=True)
class TrainingState:
    """Combined state of all phases."""
    needs_12keys_preproc: bool
    phase1: PhaseState
    phase2: PhaseState


def _has_12keys_pkl(data_root: Path, pkl_subdir: str) -> bool:
    """Check 12-keys preprocessing completed (`.done` marker present).

    pkl_subdir is the canonical folder name (e.g. 'instance_pkl_16bars_str4_fpb16_48p_12keys'),
    computed by training.pkl_paths.pkl_dir_name. State detection has no opinion
    on which config — caller passes it in.
    """
    pkl_root = data_root / "pkl_files" / pkl_subdir
    return (pkl_root / ".done").is_file()


_TRAINING_ARTEFACTS_SUBDIR = "training_artefacts"


def detect_state(data_root: Path, result_root: Path, max_epoch: int, pkl_subdir_12keys: str) -> TrainingState:
    """Top-level state detection — preproc + Phase 1 + Phase 2."""
    artefacts_root = result_root / _TRAINING_ARTEFACTS_SUBDIR
    return TrainingState(
        needs_12keys_preproc=not _has_12keys_pkl(data_root, pkl_subdir_12keys),
        phase1=detect_phase_state(artefacts_root / "idx001", max_epoch),
        phase2=detect_phase_state(artefacts_root / "idx002", max_epoch),
    )
