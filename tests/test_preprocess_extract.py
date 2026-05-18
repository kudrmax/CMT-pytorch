"""Characterization tests for preprocess.make_instance_pkl_files.

Goal: lock down the exact pkl content produced by the current implementation,
so that the upcoming extract-method refactor can be verified mechanically —
every byte of every pkl must remain identical pre- and post-refactor.

Each fixture covers a different code path inside the per-window loop:
  A. Happy path (all filters pass)
  B. Rhythm filter triggered (>75% rest in window)
  C. Chord filter triggered (<4 chord notes)
  D. Empty-onset guard (our patch — windows with only sustained notes)
  E. Pitch-range overflow (highest - base_note >= pitch_range)
  F. Multi-window file (sliding window with stride)
  G. shift=True mode (12 pitch-shifted pkl per window)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TESTS_ROOT))

import preprocess  # noqa: E402

from conftest import build_two_track_midi, write_two_track_midi_to_disk  # noqa: E402


def _hash_pkl_dir(pkl_root: Path) -> dict[str, str]:
    """Return {relative_pkl_path: md5(content)} for every .pkl under pkl_root."""
    result = {}
    if not pkl_root.exists():
        return result
    for path in sorted(pkl_root.rglob("*.pkl")):
        rel = path.relative_to(pkl_root).as_posix()
        with open(path, "rb") as f:
            result[rel] = hashlib.md5(f.read()).hexdigest()
    return result


def _run_preprocess_on_midis(tmp_path: Path, midis: list, **kwargs) -> dict[str, str]:
    """Place midis on disk, run make_instance_pkl_files with kwargs, return pkl hash dict."""
    for song_title, filename, pm in midis:
        write_two_track_midi_to_disk(pm, str(tmp_path), song_title, filename)

    defaults = dict(
        root_dir=str(tmp_path),
        midi_dir="midi",
        num_bars=8,
        frame_per_bar=16,
        stride_bars=4,
        pitch_range=48,
        shift=False,
    )
    defaults.update(kwargs)
    preprocess.make_instance_pkl_files(**defaults)

    pkl_root = tmp_path / "pkl_files"
    return _hash_pkl_dir(pkl_root)


# ---------- Fixture A: happy path ----------

def _fixture_a_happy_path():
    """9-bar C major scale (eighth notes) + I-vi-IV-V loop.

    Why 9 bars and not 8: preprocess.py's main loop is
        for i in range(0, timelen - (instance_len + 1), stride)
    A MIDI of exactly 8 bars at 120 BPM gives timelen = 129, so the loop
    executes 0 times → no pkl. We need timelen >= 130, i.e. at least one
    extra frame past the 8-bar mark. 9 bars (timelen ~= 145) gives one
    iteration at i=0 with stride=4 bars (loop = range(0, 16, 64) = [0]),
    producing exactly one pkl per shift.

    All preprocess filters pass: chord track has plenty of notes,
    rhythm has 75%+ activity, pitch span is within 4 octaves.
    """
    pitches_ascending = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5 in C major
    melody = []
    for i in range(72):  # 9 bars × 8 eighth notes
        start = i * 0.25
        end = start + 0.25
        melody.append((start, end, pitches_ascending[i % 8]))

    chord_progression = [
        [36, 48, 52, 55],   # C: C2 bass + C3 E3 G3
        [33, 45, 48, 52],   # Am: A1 bass + A2 C3 E3
        [29, 41, 45, 48],   # F: F1 bass + F2 A2 C3
        [31, 43, 47, 50],   # G: G1 bass + G2 B2 D3
    ]
    chord_groups = []
    for bar_idx in range(9):
        chord = chord_progression[bar_idx % 4]
        bar_start = bar_idx * 2.0  # 1 bar at 120 BPM = 2.0s
        chord_groups.append((bar_start, chord))

    return build_two_track_midi(melody, chord_groups)


EXPECTED_FIXTURE_A: dict[str, str] | None = {
    "instance_pkl_8bars_str4_fpb16_48p_ckey/train/song_a/song_a_00_+0_00.pkl":
        "0c30b6b24a1545f87ef111c7387b4264",
}


def test_fixture_a_happy_path(tmp_path):
    pm = _fixture_a_happy_path()
    hashes = _run_preprocess_on_midis(tmp_path, [("song_a", "song_a", pm)])
    if EXPECTED_FIXTURE_A is None:
        pytest.fail(f"capture: EXPECTED_FIXTURE_A = {hashes!r}")
    assert hashes == EXPECTED_FIXTURE_A


# ---------- Fixture B: rhythm filter triggered (>75% rest) ----------

def _fixture_b_too_much_rest():
    """One short melody note at start, then silence for 16+ seconds.

    Window 0 has rhythm_idx > 0 only in frames 0..3 (one quarter note);
    rhythm_idx.nonzero().size = 4 < instance_len // 4 = 32 → continue.
    Phantom note at the end pushes timelen past 130 so the main loop fires.
    """
    melody = [
        (0.0, 0.5, 60),       # one C4 quarter note at start
        (16.5, 17.0, 60),     # phantom note past the window to extend MIDI duration
    ]
    chord_progression = [
        [36, 48, 52, 55],
        [33, 45, 48, 52],
        [29, 41, 45, 48],
        [31, 43, 47, 50],
    ]
    chord_groups = []
    for bar_idx in range(9):
        chord = chord_progression[bar_idx % 4]
        chord_groups.append((bar_idx * 2.0, chord))
    return build_two_track_midi(melody, chord_groups)


EXPECTED_FIXTURE_B: dict[str, str] | None = {}


def test_fixture_b_rhythm_filter(tmp_path):
    pm = _fixture_b_too_much_rest()
    hashes = _run_preprocess_on_midis(tmp_path, [("song_b", "song_b", pm)])
    if EXPECTED_FIXTURE_B is None:
        pytest.fail(f"capture: EXPECTED_FIXTURE_B = {hashes!r}")
    assert hashes == EXPECTED_FIXTURE_B


# ---------- Fixture C: chord filter triggered (<4 chord notes) ----------

def _fixture_c_too_few_chord_notes():
    """Healthy 9-bar melody but chord track has only 3 notes total.

    `len(chord_inst.nonzero()[1]) < 4` → continue → 0 pkl.

    preprocess.py overrides each chord note's duration to 1 frame, so 3
    chord notes contribute exactly 3 nonzero frame-cells in chord_inst.
    """
    pitches_ascending = [60, 62, 64, 65, 67, 69, 71, 72]
    melody = [(i * 0.25, i * 0.25 + 0.25, pitches_ascending[i % 8]) for i in range(72)]
    chord_groups = [(0.0, [36, 48, 52])]  # 3 notes total — under threshold
    return build_two_track_midi(melody, chord_groups)


EXPECTED_FIXTURE_C: dict[str, str] | None = {}


def test_fixture_c_chord_filter(tmp_path):
    pm = _fixture_c_too_few_chord_notes()
    hashes = _run_preprocess_on_midis(tmp_path, [("song_c", "song_c", pm)])
    if EXPECTED_FIXTURE_C is None:
        pytest.fail(f"capture: EXPECTED_FIXTURE_C = {hashes!r}")
    assert hashes == EXPECTED_FIXTURE_C


# ---------- Fixture D: empty-onset guard (our 861bf953 patch) ----------

def _fixture_d_only_sustained_in_window():
    """16-bar MIDI with one sustained note covering everything.

    Window 0 has the onset at frame 0 (passes our guard) but fails the
    `len(set(pitch_list)) <= 5` variety filter (only 2 distinct pitch tokens:
    onset and hold).

    Window 1 (i = 64) has NO onsets in [64, 64+129) — only sustained pitches —
    triggers our `onset_inst.nonzero()[1].size == 0` guard.

    Both windows produce 0 pkl, but for different reasons. The hash dict
    will be empty either way; the snapshot guards against the refactor
    changing WHICH filter triggers (which would be a behavior change even
    if the on-disk artifact stays identical).
    """
    melody = [(0.0, 32.0, 60)]  # one note holding for 16 bars
    chord_progression = [
        [36, 48, 52, 55],
        [33, 45, 48, 52],
        [29, 41, 45, 48],
        [31, 43, 47, 50],
    ]
    chord_groups = []
    for bar_idx in range(16):
        chord = chord_progression[bar_idx % 4]
        chord_groups.append((bar_idx * 2.0, chord))
    return build_two_track_midi(melody, chord_groups)


EXPECTED_FIXTURE_D: dict[str, str] | None = {}


def test_fixture_d_empty_onset_guard(tmp_path):
    pm = _fixture_d_only_sustained_in_window()
    hashes = _run_preprocess_on_midis(tmp_path, [("song_d", "song_d", pm)])
    if EXPECTED_FIXTURE_D is None:
        pytest.fail(f"capture: EXPECTED_FIXTURE_D = {hashes!r}")
    assert hashes == EXPECTED_FIXTURE_D


# ---------- Fixture E: pitch-range overflow ----------

def _fixture_e_pitch_range_overflow():
    """Melody spans more than 4 octaves (>= pitch_range=48 semitones).

    Triggers `if highest_note - base_note >= pitch_range: continue`. The
    melody onsets cycle through 8 pitches spanning 60 semitones (C2 to C7).
    """
    melody = []
    pitches = [36, 48, 60, 72, 84, 96, 60, 72]  # span 36..96 = 60 semitones
    for i in range(72):  # 9 bars × 8 eighth notes
        start = i * 0.25
        melody.append((start, start + 0.25, pitches[i % 8]))
    chord_groups = [
        (0.0, [36, 48, 52, 55]),
        (4.0, [33, 45, 48, 52]),
        (8.0, [29, 41, 45, 48]),
        (12.0, [31, 43, 47, 50]),
    ]
    return build_two_track_midi(melody, chord_groups)


EXPECTED_FIXTURE_E: dict[str, str] | None = {}


def test_fixture_e_pitch_range_overflow(tmp_path):
    pm = _fixture_e_pitch_range_overflow()
    hashes = _run_preprocess_on_midis(tmp_path, [("song_e", "song_e", pm)])
    if EXPECTED_FIXTURE_E is None:
        pytest.fail(f"capture: EXPECTED_FIXTURE_E = {hashes!r}")
    assert hashes == EXPECTED_FIXTURE_E


# ---------- Fixture F: multi-window file (sliding window iteration) ----------

def _fixture_f_multi_window():
    """17-bar MIDI → multiple sliding-window iterations.

    timelen ≈ 273 frames, instance_len = 128, stride = 64.
    Loop = range(0, 273-129, 64) = range(0, 144, 64) = [0, 64, 128].
    Three windows; each that passes filters becomes a pkl.
    """
    pitches_ascending = [60, 62, 64, 65, 67, 69, 71, 72]
    melody = []
    for i in range(17 * 8):  # 17 bars × 8 eighth notes
        start = i * 0.25
        melody.append((start, start + 0.25, pitches_ascending[i % 8]))
    chord_progression = [
        [36, 48, 52, 55],
        [33, 45, 48, 52],
        [29, 41, 45, 48],
        [31, 43, 47, 50],
    ]
    chord_groups = []
    for bar_idx in range(17):
        chord = chord_progression[bar_idx % 4]
        chord_groups.append((bar_idx * 2.0, chord))
    return build_two_track_midi(melody, chord_groups)


EXPECTED_FIXTURE_F: dict[str, str] | None = {
    "instance_pkl_8bars_str4_fpb16_48p_ckey/train/song_f/song_f_00_+0_00.pkl":
        "0c30b6b24a1545f87ef111c7387b4264",
    "instance_pkl_8bars_str4_fpb16_48p_ckey/train/song_f/song_f_00_+0_01.pkl":
        "0c30b6b24a1545f87ef111c7387b4264",
    "instance_pkl_8bars_str4_fpb16_48p_ckey/train/song_f/song_f_00_+0_02.pkl":
        "0c30b6b24a1545f87ef111c7387b4264",
}


def test_fixture_f_multi_window(tmp_path):
    pm = _fixture_f_multi_window()
    hashes = _run_preprocess_on_midis(tmp_path, [("song_f", "song_f", pm)])
    if EXPECTED_FIXTURE_F is None:
        pytest.fail(f"capture: EXPECTED_FIXTURE_F = {hashes!r}")
    assert hashes == EXPECTED_FIXTURE_F


# ---------- Fixture G: shift=True (12 pitch-shifted pkl per window) ----------

def _fixture_g_shift_mode():
    """Same MIDI as fixture A; processed with shift=True → 12 pkl per window
    (pitch_shift = range(-5, 7) = 12 values: -5..-1, 0, +1..+6)."""
    return _fixture_a_happy_path()


EXPECTED_FIXTURE_G: dict[str, str] | None = {
    "instance_pkl_8bars_str4_fpb16_48p_12keys/train/song_g/song_g_00_+0_00.pkl": "0c30b6b24a1545f87ef111c7387b4264",
    "instance_pkl_8bars_str4_fpb16_48p_12keys/train/song_g/song_g_00_+1_00.pkl": "169b88a2a3f266b2927df8918a4516a2",
    "instance_pkl_8bars_str4_fpb16_48p_12keys/train/song_g/song_g_00_+2_00.pkl": "2e5b2917d67fb9929ef536c803a10172",
    "instance_pkl_8bars_str4_fpb16_48p_12keys/train/song_g/song_g_00_+3_00.pkl": "f64b5d2eb3956e2c5abbd240f5a6e8f9",
    "instance_pkl_8bars_str4_fpb16_48p_12keys/train/song_g/song_g_00_+4_00.pkl": "70c090354e490db9a63c8683306171c5",
    "instance_pkl_8bars_str4_fpb16_48p_12keys/train/song_g/song_g_00_+5_00.pkl": "7095c2db89fc40b3f595ae04674b9c49",
    "instance_pkl_8bars_str4_fpb16_48p_12keys/train/song_g/song_g_00_+6_00.pkl": "32251ddeb90685cdd9ac7ff72fbf96a3",
    "instance_pkl_8bars_str4_fpb16_48p_12keys/train/song_g/song_g_00_-1_00.pkl": "9ea2e12d38741bd8714cd9318c8282be",
    "instance_pkl_8bars_str4_fpb16_48p_12keys/train/song_g/song_g_00_-2_00.pkl": "adf9a1b540774ebd8d89066bfeb1fc51",
    "instance_pkl_8bars_str4_fpb16_48p_12keys/train/song_g/song_g_00_-3_00.pkl": "ef1c3b1f9f2c0901a1272e60e3ea185a",
    "instance_pkl_8bars_str4_fpb16_48p_12keys/train/song_g/song_g_00_-4_00.pkl": "790898639038cc9c7fdb6e1106d2f79f",
    "instance_pkl_8bars_str4_fpb16_48p_12keys/train/song_g/song_g_00_-5_00.pkl": "302601787011651e105de71b941f5c69",
}


def test_fixture_g_shift_mode(tmp_path):
    pm = _fixture_g_shift_mode()
    hashes = _run_preprocess_on_midis(
        tmp_path,
        [("song_g", "song_g", pm)],
        shift=True,
    )
    if EXPECTED_FIXTURE_G is None:
        pytest.fail(f"capture: EXPECTED_FIXTURE_G = {hashes!r}")
    assert hashes == EXPECTED_FIXTURE_G
