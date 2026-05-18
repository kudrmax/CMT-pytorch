"""Convert wjazzd.db (SQLite) → CMT-compatible 2-track MIDI files.

Pipeline:
    wjazzd.db  →  load_solo(melid)  →  SoloData
    SoloData   →  build_solo_midi   →  pretty_midi.PrettyMIDI
    PrettyMIDI →  save              →  midi/<solo_id>/<solo_id>.mid

The output MIDI is intended as input to authorial preprocess.py (NOT modified).
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import pretty_midi

from jazz.wjazzd.dataset_converter.filter import SoloMeta, should_skip


@dataclass(frozen=True)
class Note:
    """One melody note from `melody` table (post-conversion to seconds @ 120 BPM)."""

    onset_sec: float
    duration_sec: float
    pitch: int  # MIDI pitch (0-127)


@dataclass(frozen=True)
class Beat:
    """One beat from `beats` table."""

    bar: int
    beat: int
    chord: str           # may be empty or 'NC' — handled by forward-fill later
    bass_pitch: int | None
    signature: str       # may be empty


@dataclass(frozen=True)
class SoloData:
    """All data for a single solo, ready for MIDI assembly."""

    meta: SoloMeta
    performer: str
    title: str
    notes: tuple[Note, ...]
    beats: tuple[Beat, ...]


def _beat_to_sec(bar: int, beat: int, tatum: int, division: int) -> float:
    """Position (bar, beat, tatum) → seconds at 120 BPM, 4/4 time signature.

    Uses bar/beat/tatum/division (not melody.onset in seconds), for deterministic
    quantization compatible with authorial preprocess.py (which assumes 120 BPM).
    """
    bar = int(bar)
    beat = int(beat)
    tatum = int(tatum)
    division = max(int(division), 1)  # safeguard against malformed rows
    beats_total = bar * 4 + beat + tatum / division
    return beats_total * 0.5  # 120 BPM => 0.5 sec/beat


def _raw_to_note(
    bar: int,
    beat: int,
    tatum: int,
    division: int,
    period: float,
    pitch: int,
    duration: float,
) -> Note:
    """Build Note from raw melody table row."""
    return Note(
        onset_sec=_beat_to_sec(bar, beat, tatum, division),
        duration_sec=float(duration),
        pitch=int(pitch),
    )


def chord_changes(beats: list[Beat]) -> list[tuple[int, int, str, int]]:
    """Walk beats list, forward-fill empty/NC, emit (bar, beat, chord, bass) on changes only.

    Returns list of (bar, beat, chord_str, bass_pitch) — one entry per chord change.
    """
    out: list[tuple[int, int, str, int]] = []
    current_chord: str | None = None
    for b in beats:
        c = b.chord
        if c == "" or c == "NC":
            continue  # forward-fill: keep current_chord
        if c != current_chord:
            current_chord = c
            bass = b.bass_pitch if b.bass_pitch is not None else 36
            out.append((b.bar, b.beat, c, bass))
    return out


def load_solo(db_path: Path, melid: int) -> SoloData:
    """Load all data for one solo by melid. Raises ValueError if not found."""
    conn = sqlite3.connect(db_path)
    try:
        # Solo metadata
        cur = conn.execute(
            """
            SELECT performer, title, signature
            FROM solo_info WHERE melid = ?
            """,
            (melid,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"melid={melid} not found in solo_info")
        performer, title, signature = row

        # Bar count + signature aggregation
        cur = conn.execute(
            """
            SELECT MIN(bar), MAX(bar) FROM beats WHERE melid = ?
            """,
            (melid,),
        )
        bar_min, bar_max = cur.fetchone()
        n_bars = (bar_max - bar_min + 1) if bar_max is not None else 0

        cur = conn.execute(
            """
            SELECT DISTINCT signature FROM beats
            WHERE melid = ? AND signature IS NOT NULL
            """,
            (melid,),
        )
        beat_sigs: set[str] = {r[0] for r in cur.fetchall()}

        meta = SoloMeta(
            melid=melid,
            signature=signature or "",
            n_bars=n_bars,
            beat_signatures=beat_sigs,
        )

        # Build melody notes — onset is computed from bar/beat/tatum at 120 BPM.
        cur = conn.execute(
            """
            SELECT bar, beat, tatum, division, period, pitch, duration
            FROM melody WHERE melid = ? ORDER BY onset
            """,
            (melid,),
        )
        notes = tuple(_raw_to_note(*r) for r in cur.fetchall())

        # Beats
        cur = conn.execute(
            """
            SELECT bar, beat, chord, bass_pitch, signature
            FROM beats WHERE melid = ? ORDER BY bar, beat
            """,
            (melid,),
        )
        beats = tuple(
            Beat(
                bar=r[0],
                beat=r[1],
                chord=r[2] or "",
                bass_pitch=r[3],
                signature=r[4] or "",
            )
            for r in cur.fetchall()
        )

        return SoloData(
            meta=meta,
            performer=performer,
            title=title,
            notes=notes,
            beats=beats,
        )
    finally:
        conn.close()


from jazz.wjazzd.dataset_converter.chord_parser import ChordParseError, parse_chord

CHORD_TONE_OCTAVE_BASE = 60   # C4 — chord-tones placed in MIDI 60-71
CHORD_NOTE_DURATION_SEC = 0.5  # Quarter note at 120 BPM; preprocess will trim to 1 frame


def build_chord_track(changes: list[tuple[int, int, str, int]]) -> pretty_midi.Instrument:
    """Build pretty_midi.Instrument with bass + chord-tones at each chord change.

    Per Решение 13: bass forced into MIDI 36-47 (C2-B2) octave to guarantee
    bass < chord-tones (60+). Chord-tones placed in 60-71 octave.
    Per Решение 8: onset only on chord change (forward-fill applied upstream).
    """
    inst = pretty_midi.Instrument(program=0, name="chord")
    for bar, beat, chord_str, bass_pitch in changes:
        try:
            parsed = parse_chord(chord_str)
        except ChordParseError as e:
            print(f"  WARN: cannot parse chord {chord_str!r} at bar={bar} beat={beat}: {e}")
            continue
        onset = _beat_to_sec(bar, beat, 0, 12)
        end = onset + CHORD_NOTE_DURATION_SEC
        # Bass note (first; preprocess.py takes [1:] to skip).
        # Force into MIDI 36-47 octave (C2-B2) to guarantee bass < chord-tones (60+).
        bass_pc = int(bass_pitch) % 12
        bass_midi = 36 + bass_pc
        inst.notes.append(pretty_midi.Note(
            velocity=80, pitch=bass_midi, start=onset, end=end,
        ))
        # Chord-tones in C4-B4 octave
        for pc in sorted(parsed.pitch_classes):
            inst.notes.append(pretty_midi.Note(
                velocity=80,
                pitch=CHORD_TONE_OCTAVE_BASE + pc,
                start=onset,
                end=end,
            ))
    return inst


def build_melody_track(notes: tuple[Note, ...]) -> pretty_midi.Instrument:
    """Build pretty_midi.Instrument with melody notes. Velocity = 100, program = 0 (piano)."""
    inst = pretty_midi.Instrument(program=0, name="melody")
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=100,
            pitch=n.pitch,
            start=n.onset_sec,
            end=n.onset_sec + n.duration_sec,
        ))
    return inst


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_name(s: str) -> str:
    """Make filesystem-safe filename component."""
    return _SAFE_NAME_RE.sub("_", s)


def _solo_id(solo: SoloData) -> str:
    """Directory + filename stem for one solo."""
    perf = _safe_name(solo.performer)
    title = _safe_name(solo.title)
    return f"{solo.meta.melid:03d}_{perf}_{title}_Solo"


def build_solo_midi(solo: SoloData, out_dir: Path) -> Path:
    """Build full 2-track MIDI for solo, save to out_dir/<solo_id>/<solo_id>.mid.

    Returns path to saved MIDI.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.instruments.append(build_melody_track(solo.notes))
    pm.instruments.append(build_chord_track(chord_changes(list(solo.beats))))

    sid = _solo_id(solo)
    target_dir = out_dir / sid
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{sid}.mid"
    pm.write(str(target_path))
    return target_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "wjazzd.db",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "midi",
    )
    parser.add_argument("--limit", type=int, default=None, help="convert at most N solos (debug)")
    args = parser.parse_args(argv)

    if not args.db.is_file():
        print(f"ERROR: db not found at {args.db}; run download.py first", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    melids = [r[0] for r in conn.execute("SELECT melid FROM solo_info ORDER BY melid").fetchall()]
    conn.close()

    if args.limit is not None:
        melids = melids[: args.limit]

    converted = 0
    skipped = 0
    failed = 0
    for melid in melids:
        try:
            solo = load_solo(args.db, melid)
        except Exception as e:
            print(f"  FAIL load melid={melid}: {e}", file=sys.stderr)
            failed += 1
            continue
        skip, reason = should_skip(solo.meta)
        if skip:
            print(f"  SKIP melid={melid} ({solo.performer} - {solo.title}): {reason}")
            skipped += 1
            continue
        try:
            path = build_solo_midi(solo, args.out)
            print(f"  OK   melid={melid:3d} -> {path.name}")
            converted += 1
        except Exception as e:
            print(f"  FAIL melid={melid}: {e}", file=sys.stderr)
            failed += 1

    print()
    print(f"Total solos: {len(melids)}")
    print(f"Converted:   {converted}")
    print(f"Skipped:     {skipped}")
    print(f"Failed:      {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
