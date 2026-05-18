"""Tests for training.train command construction (subprocess calls mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from training.train import run_preprocess_shift


def test_run_preprocess_shift_command(tmp_path: Path) -> None:
    """Verify subprocess.run is called with correct args for --shift preprocess."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    midi_dir = data_root / "midi"
    midi_dir.mkdir()
    with patch("training.train.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        run_preprocess_shift(
            data_root,
            num_bars=8, stride_bars=4, frame_per_bar=16, pitch_range=48,
            pkl_subdir="instance_pkl_8bars_str4_fpb16_48p_12keys",
        )
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert "preprocess.py" in " ".join(cmd)
    assert "--shift" in cmd
    assert "--root_dir" in cmd
    assert str(data_root) in cmd
    assert "--midi_dir" in cmd
    assert "midi" in cmd
    assert "--stride_bars" in cmd
    assert "4" in cmd
    assert kwargs.get("check") is True


def test_run_preprocess_raises_if_data_missing(tmp_path: Path) -> None:
    """If data_root doesn't exist, raise FileNotFoundError before subprocess."""
    import pytest
    nonexistent = tmp_path / "no_such"
    with pytest.raises(FileNotFoundError):
        run_preprocess_shift(
            nonexistent,
            num_bars=8, stride_bars=4, frame_per_bar=16, pitch_range=48,
            pkl_subdir="instance_pkl_8bars_str4_fpb16_48p_12keys",
        )


from training.train import run_phase_1


def test_run_phase_1_fresh_command(tmp_path: Path) -> None:
    """Fresh start (no restore_epoch): builds command with --idx 1, no --restore_epoch."""
    hparams = tmp_path / "hparams_jazz.yaml"
    hparams.touch()
    with patch("training.train.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        run_phase_1(hparams_path=hparams, restore_epoch=None)
    cmd = mock_run.call_args[0][0]
    assert "run.py" in " ".join(cmd)
    assert "--idx" in cmd
    assert "1" in cmd
    assert "--hparams" in cmd
    assert str(hparams) in cmd
    assert "--restore_epoch" not in cmd


def test_run_phase_1_resume_command(tmp_path: Path) -> None:
    """Resume from checkpoint: includes --restore_epoch."""
    hparams = tmp_path / "hparams_jazz.yaml"
    hparams.touch()
    with patch("training.train.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        run_phase_1(hparams_path=hparams, restore_epoch=30)
    cmd = mock_run.call_args[0][0]
    assert "--restore_epoch" in cmd
    assert "30" in cmd


from training.train import run_phase_2


def test_run_phase_2_fresh_command(tmp_path: Path) -> None:
    """Phase 2 fresh: --idx 2 + --load_rhythm + custom hparams from result/idx002/."""
    hparams = tmp_path / "result" / "idx002" / "hparams.yaml"
    hparams.parent.mkdir(parents=True)
    hparams.touch()
    with patch("training.train.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        run_phase_2(hparams_path=hparams, restore_epoch=None)
    cmd = mock_run.call_args[0][0]
    assert "--idx" in cmd
    assert "2" in cmd
    assert "--load_rhythm" in cmd
    assert "--hparams" in cmd
    assert str(hparams) in cmd
    assert "--restore_epoch" not in cmd


def test_run_phase_2_resume_command(tmp_path: Path) -> None:
    """Phase 2 resume includes --restore_epoch but still --load_rhythm."""
    hparams = tmp_path / "result" / "idx002" / "hparams.yaml"
    hparams.parent.mkdir(parents=True)
    hparams.touch()
    with patch("training.train.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        run_phase_2(hparams_path=hparams, restore_epoch=50)
    cmd = mock_run.call_args[0][0]
    assert "--load_rhythm" in cmd
    assert "--restore_epoch" in cmd
    assert "50" in cmd


from training.train import finalize, OVERFIT_EPOCH_THRESHOLD


def test_finalize_creates_artifacts(tmp_path: Path) -> None:
    """finalize copies best PD checkpoint + writes JSON summaries."""
    result_root = tmp_path / "result"
    data_root = tmp_path / "data"
    ckpt = result_root / "training_artefacts" / "idx002" / "model" / "checkpoint_70.pth.tar"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"fake-weights")
    for split, count in [("train", 3), ("eval", 1), ("test", 1)]:
        for i in range(count):
            d = data_root / "pkl_files" / "instance_pkl_8bars_str4_fpb16_48p_ckey" / split / f"Solo_{split}_{i}"
            d.mkdir(parents=True)
            (d / "x.pkl").touch()
    finalize(
        result_root=result_root,
        data_root=data_root,
        best_pd_epoch=70,
        best_rd_epoch=80,
        max_epoch=100,
        best_rhythm_acc=0.85,
        best_pitch_acc=0.45,
        pkl_subdir_1key="instance_pkl_8bars_str4_fpb16_48p_ckey",
        final_model_name="best_jazz_model_8bars.pth.tar",
        hparams_path=Path("hparams_jazz_8bars.yaml"),
    )
    assert (result_root / "best_jazz_model_8bars.pth.tar").is_file()
    assert (result_root / "best_jazz_model_8bars.pth.tar").read_bytes() == b"fake-weights"
    import json
    summary = json.loads((result_root / "summary.json").read_text())
    assert summary["phase1_best_epoch"] == 80
    assert summary["phase1_best_rhythm_acc"] == 0.85
    assert summary["phase2_best_epoch"] == 70
    assert summary["phase2_best_pitch_acc_onset_only"] == 0.45
    assert summary["overfit_warning"] is False
    assert summary["hparams_used"] == "hparams_jazz_8bars.yaml"
    split_info = json.loads((result_root / "split_info.json").read_text())
    assert split_info["instances"]["train"] == 3
    assert "Solo_train_0" in split_info["solos"]["train"]


def test_run_preprocess_shift_touches_done_marker(tmp_path: Path) -> None:
    """After subprocess returns 0, .done marker is created in 12keys pkl dir."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    midi_dir = data_root / "midi"
    midi_dir.mkdir()
    pkl_root = data_root / "pkl_files" / "instance_pkl_8bars_str4_fpb16_48p_12keys"
    pkl_root.mkdir(parents=True)  # mock subprocess effect: pkl_root exists post-run

    with patch("training.train.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        run_preprocess_shift(
            data_root,
            num_bars=8, stride_bars=4, frame_per_bar=16, pitch_range=48,
            pkl_subdir="instance_pkl_8bars_str4_fpb16_48p_12keys",
        )
    assert (pkl_root / ".done").is_file(), "preproc success should create .done marker"


def test_run_preprocess_shift_no_done_on_subprocess_failure(tmp_path: Path) -> None:
    """If subprocess raises (check=True), .done is NOT written."""
    import pytest
    import subprocess as sp_module
    data_root = tmp_path / "data"
    data_root.mkdir()
    midi_dir = data_root / "midi"
    midi_dir.mkdir()
    pkl_root = data_root / "pkl_files" / "instance_pkl_8bars_str4_fpb16_48p_12keys"
    pkl_root.mkdir(parents=True)

    with patch("training.train.subprocess.run") as mock_run:
        mock_run.side_effect = sp_module.CalledProcessError(returncode=1, cmd=["preprocess.py"])
        with pytest.raises(sp_module.CalledProcessError):
            run_preprocess_shift(
                data_root,
                num_bars=8, stride_bars=4, frame_per_bar=16, pitch_range=48,
                pkl_subdir="instance_pkl_8bars_str4_fpb16_48p_12keys",
            )
    assert not (pkl_root / ".done").exists(), ".done must NOT be created on failure"


def test_finalize_overfit_signal(tmp_path: Path) -> None:
    """best_pd_epoch < threshold → overfit_warning=True."""
    result_root = tmp_path / "result"
    data_root = tmp_path / "data"
    ckpt = result_root / "training_artefacts" / "idx002" / "model" / "checkpoint_20.pth.tar"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    for split in ("train", "eval", "test"):
        d = data_root / "pkl_files" / "instance_pkl_8bars_str4_fpb16_48p_ckey" / split / "X"
        d.mkdir(parents=True)
        (d / "y.pkl").touch()
    finalize(
        result_root=result_root,
        data_root=data_root,
        best_pd_epoch=20,
        best_rd_epoch=80,
        max_epoch=100,
        best_rhythm_acc=0.85,
        best_pitch_acc=0.45,
        pkl_subdir_1key="instance_pkl_8bars_str4_fpb16_48p_ckey",
        final_model_name="best_jazz_model_8bars.pth.tar",
        hparams_path=Path("hparams_jazz_8bars.yaml"),
    )
    import json
    summary = json.loads((result_root / "summary.json").read_text())
    assert summary["overfit_warning"] is True
