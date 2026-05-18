"""Tests for training.pkl_paths."""
from __future__ import annotations

import pytest

from training.pkl_paths import pkl_dir_name, pkl_preprocess_params


def test_pkl_dir_name_ckey():
    assert pkl_dir_name(num_bars=8, stride_bars=4, frame_per_bar=16, pitch_range=48, shift=False) == "instance_pkl_8bars_str4_fpb16_48p_ckey"


def test_pkl_dir_name_12keys():
    assert pkl_dir_name(num_bars=8, stride_bars=4, frame_per_bar=16, pitch_range=48, shift=True) == "instance_pkl_8bars_str4_fpb16_48p_12keys"


def test_pkl_dir_name_16bars():
    assert pkl_dir_name(num_bars=16, stride_bars=4, frame_per_bar=16, pitch_range=48, shift=False) == "instance_pkl_16bars_str4_fpb16_48p_ckey"


def test_pkl_dir_name_16bars_auto_stride():
    assert pkl_dir_name(num_bars=16, stride_bars=8, frame_per_bar=16, pitch_range=48, shift=False) == "instance_pkl_16bars_str8_fpb16_48p_ckey"


def test_pkl_preprocess_params_extracts_all_four():
    hp = {
        "data_io": {
            "preprocess": {
                "num_bars": 16,
                "stride_bars": 4,
                "frame_per_bar": 16,
                "pitch_range": 48,
            }
        }
    }
    nb, sb, fpb, pr = pkl_preprocess_params(hp)
    assert (nb, sb, fpb, pr) == (16, 4, 16, 48)


def test_pkl_preprocess_params_missing_section_raises():
    with pytest.raises(KeyError, match="data_io.preprocess"):
        pkl_preprocess_params({"data_io": {}})


def test_pkl_preprocess_params_missing_field_raises():
    hp = {"data_io": {"preprocess": {"num_bars": 16, "stride_bars": 4, "frame_per_bar": 16}}}
    with pytest.raises(KeyError, match="pitch_range"):
        pkl_preprocess_params(hp)
