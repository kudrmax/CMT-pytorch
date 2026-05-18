"""Tests for training.resume_detector."""
from __future__ import annotations

from pathlib import Path

from training.resume_detector import detect_phase_state, PhaseState, detect_state, TrainingState


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_phase_not_started(tmp_path: Path) -> None:
    asset_path = tmp_path / "idx001"
    state = detect_phase_state(asset_path, max_epoch=100)
    assert state.status == "not_started"
    assert state.last_epoch == 0


def test_phase_partial(tmp_path: Path) -> None:
    asset_path = tmp_path / "idx001"
    _touch(asset_path / "model" / "checkpoint_10.pth.tar")
    _touch(asset_path / "model" / "checkpoint_20.pth.tar")
    _touch(asset_path / "model" / "checkpoint_30.pth.tar")
    state = detect_phase_state(asset_path, max_epoch=100)
    assert state.status == "partial"
    assert state.last_epoch == 30


def test_phase_complete(tmp_path: Path) -> None:
    asset_path = tmp_path / "idx001"
    for e in [10, 20, 50, 100]:
        _touch(asset_path / "model" / f"checkpoint_{e}.pth.tar")
    state = detect_phase_state(asset_path, max_epoch=100)
    assert state.status == "complete"
    assert state.last_epoch == 100


def test_phase_partial_unsorted_filenames(tmp_path: Path) -> None:
    asset_path = tmp_path / "idx001"
    for e in [50, 10, 30, 20]:
        _touch(asset_path / "model" / f"checkpoint_{e}.pth.tar")
    state = detect_phase_state(asset_path, max_epoch=100)
    assert state.status == "partial"
    assert state.last_epoch == 50


def test_ignores_non_checkpoint_files(tmp_path: Path) -> None:
    asset_path = tmp_path / "idx001"
    _touch(asset_path / "log.txt")
    _touch(asset_path / "sampling_results" / "epoch_010" / "sample.mid")
    _touch(asset_path / "model" / "checkpoint_10.pth.tar")
    _touch(asset_path / "model" / "random_other_file.txt")
    state = detect_phase_state(asset_path, max_epoch=100)
    assert state.status == "partial"
    assert state.last_epoch == 10


from training.resume_detector import detect_state, TrainingState


def test_detect_state_fresh(tmp_path: Path) -> None:
    """Fresh: needs preprocessing, no phases started."""
    data_root = tmp_path / "data"
    result_root = tmp_path / "result"
    data_root.mkdir()
    state = detect_state(data_root, result_root, max_epoch=100,
                         pkl_subdir_12keys="instance_pkl_8bars_str4_fpb16_48p_12keys")
    assert state.needs_12keys_preproc is True
    assert state.phase1.status == "not_started"
    assert state.phase2.status == "not_started"


def test_detect_state_preproc_done_phase1_partial(tmp_path: Path) -> None:
    """12keys pkl exists, Phase 1 has 30 epochs."""
    data_root = tmp_path / "data"
    pkl_root = data_root / "pkl_files" / "instance_pkl_8bars_str4_fpb16_48p_12keys"
    pkl_dir = pkl_root / "train" / "ArtPepper_X"
    pkl_dir.mkdir(parents=True)
    (pkl_dir / "ArtPepper_X_00_+0_00.pkl").touch()
    (pkl_root / ".done").touch()   # NEW: marker for completed preprocess
    result_root = tmp_path / "result"
    _touch(result_root / "training_artefacts" / "idx001" / "model" / "checkpoint_10.pth.tar")
    _touch(result_root / "training_artefacts" / "idx001" / "model" / "checkpoint_30.pth.tar")
    state = detect_state(data_root, result_root, max_epoch=100,
                         pkl_subdir_12keys="instance_pkl_8bars_str4_fpb16_48p_12keys")
    assert state.needs_12keys_preproc is False
    assert state.phase1.status == "partial"
    assert state.phase1.last_epoch == 30
    assert state.phase2.status == "not_started"


def test_detect_state_phase2_complete(tmp_path: Path) -> None:
    """Both phases complete."""
    data_root = tmp_path / "data"
    pkl_root = data_root / "pkl_files" / "instance_pkl_8bars_str4_fpb16_48p_12keys"
    pkl_dir = pkl_root / "train" / "X"
    pkl_dir.mkdir(parents=True)
    (pkl_dir / "X_00_+0_00.pkl").touch()
    (pkl_root / ".done").touch()
    result_root = tmp_path / "result"
    for e in [10, 20, 100]:
        _touch(result_root / "training_artefacts" / "idx001" / "model" / f"checkpoint_{e}.pth.tar")
        _touch(result_root / "training_artefacts" / "idx002" / "model" / f"checkpoint_{e}.pth.tar")
    state = detect_state(data_root, result_root, max_epoch=100,
                         pkl_subdir_12keys="instance_pkl_8bars_str4_fpb16_48p_12keys")
    assert state.phase1.status == "complete"
    assert state.phase2.status == "complete"


def test_has_12keys_pkl_returns_false_without_done_marker(tmp_path: Path) -> None:
    """Pkl files exist but .done marker absent — preprocess incomplete, return False."""
    data_root = tmp_path / "data"
    pkl_dir = data_root / "pkl_files" / "instance_pkl_8bars_str4_fpb16_48p_12keys" / "train" / "X"
    pkl_dir.mkdir(parents=True)
    (pkl_dir / "x.pkl").touch()
    state = detect_state(data_root, tmp_path / "result", max_epoch=100,
                         pkl_subdir_12keys="instance_pkl_8bars_str4_fpb16_48p_12keys")
    assert state.needs_12keys_preproc is True


def test_has_12keys_pkl_returns_true_with_done_marker(tmp_path: Path) -> None:
    """Pkl files + .done marker → preproc complete."""
    data_root = tmp_path / "data"
    pkl_root = data_root / "pkl_files" / "instance_pkl_8bars_str4_fpb16_48p_12keys"
    pkl_dir = pkl_root / "train" / "X"
    pkl_dir.mkdir(parents=True)
    (pkl_dir / "x.pkl").touch()
    (pkl_root / ".done").touch()
    state = detect_state(data_root, tmp_path / "result", max_epoch=100,
                         pkl_subdir_12keys="instance_pkl_8bars_str4_fpb16_48p_12keys")
    assert state.needs_12keys_preproc is False
