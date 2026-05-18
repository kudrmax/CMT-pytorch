"""Integration tests for jazz.wjazzd.dataset_converter.wjazzd_to_cmt_midi."""
from __future__ import annotations

from pathlib import Path

import pytest

from jazz.wjazzd.dataset_converter.wjazzd_to_cmt_midi import _beat_to_sec, load_solo, SoloData


class TestBeatToSec:
    def test_zero_position(self) -> None:
        assert _beat_to_sec(0, 0, 0, 12) == pytest.approx(0.0)

    def test_one_bar(self) -> None:
        # bar=1, beat=0, tatum=0 at 4/4 = 4 beats from start = 2 seconds at 120 BPM
        assert _beat_to_sec(1, 0, 0, 12) == pytest.approx(2.0)

    def test_one_beat(self) -> None:
        # bar=0, beat=1, tatum=0 = 1 beat = 0.5 sec at 120 BPM
        assert _beat_to_sec(0, 1, 0, 12) == pytest.approx(0.5)

    def test_half_beat_via_tatum(self) -> None:
        # bar=0, beat=0, tatum=6, division=12 = 0.5 beat = 0.25 sec
        assert _beat_to_sec(0, 0, 6, 12) == pytest.approx(0.25)

    def test_triplet_position(self) -> None:
        # division=3 = triplets; tatum=1 of 3 = 1/3 of a beat
        assert _beat_to_sec(0, 0, 1, 3) == pytest.approx(0.5 / 3)

    def test_division_zero_treated_as_1(self) -> None:
        # Some malformed rows in WJazzD have division=0; treat as 1 (no tatum)
        assert _beat_to_sec(1, 2, 0, 0) == pytest.approx((4 + 2) * 0.5)


def test_load_known_solo(wjazzd_db_path: Path) -> None:
    """melid=1 = first solo in WJazzD; should have melody and beats data."""
    solo = load_solo(wjazzd_db_path, melid=1)
    assert isinstance(solo, SoloData)
    assert solo.meta.melid == 1
    assert solo.meta.signature == "4/4"  # melid=1 is 4/4 in WJazzD
    assert len(solo.notes) > 0
    assert len(solo.beats) > 0


def test_load_unknown_melid_raises(wjazzd_db_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_solo(wjazzd_db_path, melid=99999)


from jazz.wjazzd.dataset_converter.wjazzd_to_cmt_midi import chord_changes, Beat


class TestChordChanges:
    def test_no_chord_no_changes(self) -> None:
        beats = [Beat(0, 0, "", None, "4/4"), Beat(0, 1, "", None, "4/4")]
        assert chord_changes(beats) == []

    def test_single_chord(self) -> None:
        beats = [Beat(0, 0, "C", 36, "4/4"), Beat(0, 1, "", 36, "4/4")]
        result = chord_changes(beats)
        assert result == [(0, 0, "C", 36)]

    def test_change_emits_new_event(self) -> None:
        beats = [
            Beat(0, 0, "C", 36, "4/4"),
            Beat(0, 1, "", 36, "4/4"),
            Beat(0, 2, "F", 41, "4/4"),
            Beat(0, 3, "", 41, "4/4"),
        ]
        result = chord_changes(beats)
        assert result == [
            (0, 0, "C", 36),
            (0, 2, "F", 41),
        ]

    def test_repeated_chord_does_not_emit(self) -> None:
        beats = [
            Beat(0, 0, "C", 36, "4/4"),
            Beat(0, 1, "C", 36, "4/4"),  # explicit repeat — should NOT emit
        ]
        result = chord_changes(beats)
        assert len(result) == 1

    def test_NC_treated_as_fill(self) -> None:
        beats = [
            Beat(0, 0, "C", 36, "4/4"),
            Beat(0, 1, "NC", 36, "4/4"),  # NC = forward fill
            Beat(0, 2, "F", 41, "4/4"),
        ]
        result = chord_changes(beats)
        assert result == [(0, 0, "C", 36), (0, 2, "F", 41)]


import pretty_midi
from jazz.wjazzd.dataset_converter.wjazzd_to_cmt_midi import Note, build_melody_track


class TestBuildMelodyTrack:
    def test_creates_instrument(self) -> None:
        notes = (
            Note(onset_sec=0.0, duration_sec=0.5, pitch=60),
            Note(onset_sec=0.5, duration_sec=0.5, pitch=62),
        )
        inst = build_melody_track(notes)
        assert isinstance(inst, pretty_midi.Instrument)
        assert len(inst.notes) == 2
        assert inst.notes[0].pitch == 60
        assert inst.notes[0].start == pytest.approx(0.0)
        assert inst.notes[0].end == pytest.approx(0.5)

    def test_empty_notes_empty_instrument(self) -> None:
        inst = build_melody_track(())
        assert len(inst.notes) == 0


from jazz.wjazzd.dataset_converter.wjazzd_to_cmt_midi import build_chord_track


class TestBuildChordTrack:
    def test_creates_chord_instrument(self) -> None:
        # Two changes: C at beat 0, F at beat 2 of bar 0
        changes = [(0, 0, "C", 36), (0, 2, "F", 41)]
        inst = build_chord_track(changes)
        assert isinstance(inst, pretty_midi.Instrument)
        # Per change: 1 bass + 3 chord-tones for triad = 4 notes
        # C (3 tones C/E/G + bass = 4) + F (3 tones F/A/C + bass = 4) = 8 notes
        assert len(inst.notes) == 8

    def test_chord_onset_at_correct_time(self) -> None:
        # C at bar=0,beat=2 = 2 beats from start = 1.0 sec at 120 BPM
        changes = [(0, 2, "C", 36)]
        inst = build_chord_track(changes)
        # All notes for this chord start at the same time
        for note in inst.notes:
            assert note.start == pytest.approx(1.0)

    def test_bass_is_lowest_pitch(self) -> None:
        # bass should be lowest among all notes for a single chord change
        changes = [(0, 0, "C", 36)]
        inst = build_chord_track(changes)
        sorted_pitches = sorted(n.pitch for n in inst.notes)
        bass = sorted_pitches[0]
        chord_tones = sorted_pitches[1:]
        assert bass < min(chord_tones)


from jazz.wjazzd.dataset_converter.wjazzd_to_cmt_midi import build_solo_midi


class TestBuildSoloMidi:
    def test_full_solo_midi(self, wjazzd_db_path: Path, tmp_path: Path) -> None:
        """E2E on melid=1: load → build → save → reopen with pretty_midi → verify 2 instruments."""
        from jazz.wjazzd.dataset_converter.wjazzd_to_cmt_midi import load_solo
        solo = load_solo(wjazzd_db_path, melid=1)
        out_dir = tmp_path / "out"
        midi_path = build_solo_midi(solo, out_dir)
        assert midi_path.is_file()
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        assert len(pm.instruments) == 2
        assert pm.instruments[0].name == "melody"
        assert pm.instruments[1].name == "chord"
        assert len(pm.instruments[0].notes) == len(solo.notes)
