"""Utility for deriving pkl folder names from preprocess parameters.

The pkl folder encodes 4 preprocess axes — num_bars, stride_bars, frame_per_bar,
pitch_range — plus a shift flag (1-key vs 12-keys augmentation). Single source
of truth so both preprocess.py and training/train.py compute identical paths.
"""
from __future__ import annotations

from typing import Any


def pkl_dir_name(
    *,
    num_bars: int,
    stride_bars: int,
    frame_per_bar: int,
    pitch_range: int,
    shift: bool,
) -> str:
    """Return canonical name of pkl folder under jazz/wjazzd/data/pkl_files/.

    Example: pkl_dir_name(num_bars=16, stride_bars=4, frame_per_bar=16,
    pitch_range=48, shift=False) → 'instance_pkl_16bars_str4_fpb16_48p_ckey'.
    """
    suffix = "12keys" if shift else "ckey"
    return f"instance_pkl_{num_bars}bars_str{stride_bars}_fpb{frame_per_bar}_{pitch_range}p_{suffix}"


def pkl_preprocess_params(hparams: dict[str, Any]) -> tuple[int, int, int, int]:
    """Extract (num_bars, stride_bars, frame_per_bar, pitch_range) from hparams.

    Reads from hparams['data_io']['preprocess']. Raises KeyError if section
    or any of the 4 fields is missing — preprocess params are mandatory.
    """
    try:
        section = hparams["data_io"]["preprocess"]
    except KeyError as e:
        raise KeyError(f"hparams missing required section: data_io.preprocess (got {e})") from e
    try:
        return (
            int(section["num_bars"]),
            int(section["stride_bars"]),
            int(section["frame_per_bar"]),
            int(section["pitch_range"]),
        )
    except KeyError as e:
        raise KeyError(f"hparams.data_io.preprocess missing field: {e}") from e
