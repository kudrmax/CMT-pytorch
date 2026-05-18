"""Tests for training.setup_hparams."""
from __future__ import annotations

from pathlib import Path

from training.setup_hparams import load_hparams, save_hparams, derive_phase_hparams


def test_round_trip(tmp_path: Path) -> None:
    """save → load returns the same data."""
    src = tmp_path / "test.yaml"
    data = {
        "model": {"hidden_dim": 512, "num_layers": 8},
        "experiment": {"max_epoch": 100, "lr": 0.0001},
    }
    save_hparams(src, data)
    loaded = load_hparams(src)
    assert loaded == data


def test_load_real_hparams_jazz_8bars() -> None:
    """Sanity: real hparams_jazz_8bars.yaml loads with expected top-level keys + preprocess section."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    data = load_hparams(repo_root / "hparams_jazz_8bars.yaml")
    assert "model" in data
    assert "experiment" in data
    assert "optimizer" in data
    assert "data_io" in data
    assert "preprocess" in data["data_io"]
    assert data["data_io"]["preprocess"]["num_bars"] == 8


def test_load_real_hparams_jazz_16bars() -> None:
    """Sanity: real hparams_jazz_16bars.yaml loads."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    data = load_hparams(repo_root / "hparams_jazz_16bars.yaml")
    assert data["data_io"]["preprocess"]["num_bars"] == 16
    assert data["data_io"]["preprocess"]["stride_bars"] == 4


def test_derive_phase_hparams_phase1(tmp_path: Path) -> None:
    """Phase 1: inject asset_root + data_path, no restore_rhythm.epoch."""
    src = tmp_path / "src.yaml"
    save_hparams(src, {
        "asset_root": "path/to/save/results",
        "data_io": {"path": "path/to/instance/pklfiles", "loader": {"batch_size": 64}},
        "experiment": {"max_epoch": 100, "restore_rhythm": {"idx": 1, "epoch": 100}},
    })
    dst = tmp_path / "result" / "idx001" / "hparams.yaml"
    derive_phase_hparams(
        src, dst,
        asset_root=tmp_path / "result",
        data_path=tmp_path / "data" / "pkl_files",
    )
    loaded = load_hparams(dst)
    assert loaded["asset_root"] == str(tmp_path / "result")
    assert loaded["data_io"]["path"] == str(tmp_path / "data" / "pkl_files")
    assert loaded["data_io"]["loader"]["batch_size"] == 64  # preserved
    assert loaded["experiment"]["restore_rhythm"]["epoch"] == 100  # unchanged
    assert loaded["experiment"]["max_epoch"] == 100  # unchanged


def test_derive_phase_hparams_phase2(tmp_path: Path) -> None:
    """Phase 2: inject all three (asset_root, data_path, restore_rhythm.epoch)."""
    src = tmp_path / "src.yaml"
    save_hparams(src, {
        "asset_root": "x", "data_io": {"path": "y"},
        "experiment": {"restore_rhythm": {"idx": 1, "epoch": 100}},
    })
    dst = tmp_path / "result" / "idx002" / "hparams.yaml"
    derive_phase_hparams(
        src, dst,
        asset_root=tmp_path / "result",
        data_path=tmp_path / "data" / "pkl_files",
        restore_rhythm_epoch=80,
    )
    loaded = load_hparams(dst)
    assert loaded["asset_root"] == str(tmp_path / "result")
    assert loaded["data_io"]["path"] == str(tmp_path / "data" / "pkl_files")
    assert loaded["experiment"]["restore_rhythm"]["epoch"] == 80


def test_derive_phase_hparams_max_epoch_override(tmp_path: Path) -> None:
    """max_epoch_override replaces experiment.max_epoch in derived file."""
    src = tmp_path / "src.yaml"
    save_hparams(src, {
        "asset_root": "x", "data_io": {"path": "y"},
        "experiment": {"max_epoch": 100},
    })
    dst = tmp_path / "dst.yaml"
    derive_phase_hparams(
        src, dst,
        asset_root=tmp_path / "result",
        data_path=tmp_path / "data",
        max_epoch_override=5,
    )
    loaded = load_hparams(dst)
    assert loaded["experiment"]["max_epoch"] == 5


def test_derive_phase_hparams_rejects_same_path(tmp_path: Path) -> None:
    """src and dst must be different paths."""
    import pytest
    p = tmp_path / "x.yaml"
    save_hparams(p, {"asset_root": "x", "data_io": {"path": "y"}})
    with pytest.raises(ValueError, match="different paths"):
        derive_phase_hparams(p, p, asset_root=tmp_path, data_path=tmp_path)


def test_derive_phase_hparams_preserves_preprocess_section(tmp_path: Path) -> None:
    """data_io.preprocess (num_bars/stride_bars/etc.) проносится через derive."""
    src = tmp_path / "src.yaml"
    save_hparams(src, {
        "asset_root": "x",
        "data_io": {
            "path": "y",
            "preprocess": {
                "num_bars": 16,
                "stride_bars": 4,
                "frame_per_bar": 16,
                "pitch_range": 48,
            },
        },
        "experiment": {"max_epoch": 100, "restore_rhythm": {"idx": 1, "epoch": 100}},
    })
    dst = tmp_path / "dst.yaml"
    derive_phase_hparams(src, dst, asset_root=tmp_path / "r", data_path=tmp_path / "d")
    loaded = load_hparams(dst)
    assert loaded["data_io"]["preprocess"]["num_bars"] == 16
    assert loaded["data_io"]["preprocess"]["stride_bars"] == 4
