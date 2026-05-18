"""pytest fixtures and shared helpers for the test suite."""
import os
from pathlib import Path
from typing import Iterable

import pretty_midi
import pytest

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DB = REPO_ROOT / "jazz" / "wjazzd" / "data" / "wjazzd.db"


@pytest.fixture(scope="session")
def wjazzd_db_path() -> Path:
    """Path to wjazzd.db. Tests requiring real DB skip if absent."""
    if not DEFAULT_DB.is_file():
        pytest.skip(f"wjazzd.db not found at {DEFAULT_DB}; run download.py first")
    return DEFAULT_DB


def build_two_track_midi(
    melody_notes: Iterable[tuple[float, float, int]],
    chord_groups: Iterable[tuple[float, list[int]]],
    tempo_bpm: float = 120.0,
) -> pretty_midi.PrettyMIDI:
    """Build a 2-track MIDI in the format preprocess.py expects.

    Track 0: melody (single-voice notes, full durations).
    Track 1: chord (bass + chord tones emitted at each chord change; durations
             are placeholders — preprocess.py overrides them to unit_time).

    Args:
      melody_notes: iterable of (start_sec, end_sec, midi_pitch).
      chord_groups: iterable of (start_sec, [bass_midi, *chord_tones_midi]).
      tempo_bpm: must be 120 to align with preprocess.py's frame grid.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    melody = pretty_midi.Instrument(program=0, name="melody")
    chord = pretty_midi.Instrument(program=0, name="chord")

    for start, end, pitch in melody_notes:
        melody.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
        )

    chord_dur = 0.5  # placeholder; preprocess.py overrides .end to start + unit_time
    for start, midi_pitches in chord_groups:
        for p in midi_pitches:
            chord.notes.append(
                pretty_midi.Note(velocity=100, pitch=p, start=start, end=start + chord_dur)
            )

    pm.instruments.append(melody)
    pm.instruments.append(chord)
    return pm


def write_two_track_midi_to_disk(
    pm: pretty_midi.PrettyMIDI, root_dir: str, song_title: str, filename: str
) -> str:
    """Place MIDI inside the nested layout preprocess.py globs (`<midi_dir>/<song>/<file>.mid`)."""
    midi_dir = "midi"
    target = os.path.join(root_dir, midi_dir, song_title)
    os.makedirs(target, exist_ok=True)
    path = os.path.join(target, f"{filename}.mid")
    pm.write(path)
    return path
