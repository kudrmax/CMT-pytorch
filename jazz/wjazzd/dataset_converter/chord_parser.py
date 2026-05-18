"""Parser for jazzomat chord strings → MIDI pitch classes.

TODO: проверить логику парсера — пройтись по полному корпусу wjazzd.db,
сверить выходные pitch-классы с ожиданием, особенно edge-cases:
alt-доминанты, sus, slash-аккорды, нестандартные расширения.


Supports jazzomat-specific shorthand:
  - 'j' for maj7 (e.g. 'Cj7' = Cmaj7)
  - '-' for minor (e.g. 'C-7' = Cm7)
  - 'b' for flat (e.g. 'Bb' = B-flat, 'C7b9' = C7 with flat-9)
  - '#' for sharp
  - 'ø' for half-diminished (m7b5)
  - '°' for diminished
  - 'alt' for altered dominant
  - 'sus2', 'sus4' for suspended chords

NOT compatible with music21.harmony.ChordSymbol — empirical test
showed 90%+ semantic mismatch on real wjazzd.db data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


class ChordParseError(ValueError):
    """Raised when a chord string cannot be parsed."""


@dataclass(frozen=True)
class ParsedChord:
    """Result of parsing a jazzomat chord string.

    pitch_classes: set of integers in [0, 12) for each pitch class in the chord.
        Includes the root. Does NOT include slash-bass override
        (use beats.bass_pitch from WJazzD instead).
    """

    root_pc: int  # 0-11
    pitch_classes: frozenset[int]
    raw: str  # Original chord string for diagnostics


_ROOT_PC = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

_ROOT_REGEX = re.compile(r"^([A-G])(bb|##|b|#)?")


def _parse_root(chord_str: str) -> tuple[int, str]:
    """Extract root pitch class and return (root_pc, remainder)."""
    m = _ROOT_REGEX.match(chord_str)
    if m is None:
        raise ChordParseError(f"no valid root in {chord_str!r}")
    note, accidental = m.group(1), m.group(2) or ""
    pc = _ROOT_PC[note]
    if accidental == "b":
        pc = (pc - 1) % 12
    elif accidental == "bb":
        pc = (pc - 2) % 12
    elif accidental == "#":
        pc = (pc + 1) % 12
    elif accidental == "##":
        pc = (pc + 2) % 12
    return pc, chord_str[m.end():]


# Quality intervals (semitones from root). NOT including root itself.
# ORDER MATTERS — _parse_quality matches longest token first.
_QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj7":  (4, 7, 11),
    "j7":    (4, 7, 11),  # jazzomat shorthand
    "mmaj7": (3, 7, 11),
    "m7b5":  (3, 6, 10),
    "ø":     (3, 6, 10),
    "dim7":  (3, 6, 9),
    "°7":    (3, 6, 9),
    "o7":    (3, 6, 9),   # ASCII alternative for °7 (jazzomat corpus)
    "m7":    (3, 7, 10),
    "-7":    (3, 7, 10),  # jazzomat shorthand
    "m6":    (3, 7, 9),
    "-6":    (3, 7, 9),
    "dim":   (3, 6),
    "°":     (3, 6),
    "o":     (3, 6),      # ASCII alternative for ° (jazzomat corpus)
    "aug":   (4, 8),
    "+":     (4, 8),
    "7":     (4, 7, 10),
    # Natural extensions — map to same pitch classes as dominant 7 (Решение 7b)
    "13":    (4, 7, 10),
    "11":    (4, 7, 10),
    "9":     (4, 7, 10),
    "6":     (4, 7, 9),
    "m":     (3, 7),
    "-":     (3, 7),
    "":      (4, 7),       # bare letter -> major triad
}

# Keys sorted by length descending — ensures longest match wins (e.g. m7 before m).
_QUALITY_KEYS_BY_LEN = sorted(_QUALITY_INTERVALS.keys(), key=len, reverse=True)


def _parse_quality(remainder: str, root_pc: int) -> tuple[set[int], str]:
    """Parse quality token. Handles sus chords as a special case."""
    # Check sus first (sus chords are NOT in _QUALITY_INTERVALS dict)
    if remainder.startswith("sus") or remainder.startswith("7sus"):
        is_dom = remainder.startswith("7sus")
        sus_start = 1 if is_dom else 0
        # Extract sus2/sus4/sus
        rest = remainder[sus_start:]
        if rest.startswith("sus2"):
            third_iv = 2  # sus2
            sus_len = 4
        elif rest.startswith("sus4"):
            third_iv = 5  # sus4
            sus_len = 4
        elif rest.startswith("sus"):
            third_iv = 5  # bare 'sus' = sus4
            sus_len = 3
        else:
            raise ChordParseError(f"bad sus chord: {remainder!r}")
        pcs = {(root_pc + third_iv) % 12, (root_pc + 7) % 12}
        if is_dom:
            pcs.add((root_pc + 10) % 12)
        return pcs, remainder[sus_start + sus_len:]

    # Otherwise standard quality match (existing logic)
    for k in _QUALITY_KEYS_BY_LEN:
        if remainder.startswith(k):
            intervals = _QUALITY_INTERVALS[k]
            pcs = {(root_pc + iv) % 12 for iv in intervals}
            return pcs, remainder[len(k):]
    raise ChordParseError(f"unknown quality at remainder {remainder!r}")


# Alteration regex: matches standard (b|#)(\d+) form, 'alt', or bare extension numbers.
_ALT_TOKEN_RE = re.compile(r"(alt|b13|#11|b9|#9|b5|#5|b6|#6|b7|9|11|13|6|7)")

# Jazzomat reversed-order alteration: number THEN accidental (e.g. '9b' = b9, '11#' = #11).
# Only matches digits that are valid extension numbers.
_ALT_REVERSED_RE = re.compile(r"(9|11|13)(b|#)")

_ALT_INTERVALS: dict[str, tuple[int | None, int | None]] = {
    # alteration_token -> (interval_to_add, interval_to_remove or None)
    "b9":  (1, None),
    "#9":  (3, None),
    "#11": (6, None),
    "b13": (8, None),
    "b5":  (6, 7),    # replaces natural 5
    "#5":  (8, 7),    # replaces natural 5
    "b6":  (8, 9),    # rare; replaces natural 6 if present
    "#6":  (10, 9),
    "b7":  (10, None),
    # Natural extensions: NOT added per Решение 7b
    "9":   (None, None),
    "11":  (None, None),
    "13":  (None, None),
    "6":   (None, None),
    "7":   (None, None),
}


def _parse_alterations(remainder: str, pcs: set[int], root_pc: int) -> tuple[set[int], str]:
    """Apply alteration tokens, return (new_pcs, leftover_remainder)."""
    while remainder:
        if remainder.startswith("alt"):
            for ad in (6, 8, 1, 3):
                pcs.add((root_pc + ad) % 12)
            pcs.discard((root_pc + 7) % 12)  # alt removes natural 5
            remainder = remainder[3:]
            continue
        # Jazzomat reversed form checked FIRST: digit THEN accidental (e.g. '9b'=b9, '11#'=#11).
        # Must come before standard form so '9b' is not greedily consumed as bare '9' + leftover 'b'.
        mr = _ALT_REVERSED_RE.match(remainder)
        if mr is not None:
            canonical = mr.group(2) + mr.group(1)  # e.g. 'b' + '9' = 'b9'
            add_iv, remove_iv = _ALT_INTERVALS.get(canonical, (None, None))
            if remove_iv is not None:
                pcs.discard((root_pc + remove_iv) % 12)
            if add_iv is not None:
                pcs.add((root_pc + add_iv) % 12)
            remainder = remainder[mr.end():]
            continue
        # Standard form: (b|#)digit or bare digit
        m = _ALT_TOKEN_RE.match(remainder)
        if m is not None:
            token = m.group(1)
            add_iv, remove_iv = _ALT_INTERVALS[token]
            if remove_iv is not None:
                pcs.discard((root_pc + remove_iv) % 12)
            if add_iv is not None:
                pcs.add((root_pc + add_iv) % 12)
            remainder = remainder[m.end():]
            continue
        break
    return pcs, remainder


def parse_chord(chord_str: str) -> ParsedChord:
    """Parse jazzomat chord string into MIDI pitch classes.

    Raises ChordParseError on empty, NC, or unrecognized syntax.
    """
    if not chord_str or chord_str in ("NC", "N.C."):
        raise ChordParseError(f"no-chord marker: {chord_str!r}")
    # Strip slash bass (e.g. 'C/E' -> 'C'). Bass pitch comes from beats.bass_pitch in WJazzD.
    if "/" in chord_str:
        chord_str_no_slash = chord_str.split("/", 1)[0]
    else:
        chord_str_no_slash = chord_str
    root_pc, remainder = _parse_root(chord_str_no_slash)
    chord_pcs, remainder = _parse_quality(remainder, root_pc)
    pcs = {root_pc} | chord_pcs
    pcs, remainder = _parse_alterations(remainder, pcs, root_pc)
    if remainder:
        raise ChordParseError(
            f"unparsed trailing tokens in {chord_str!r}: {remainder!r}"
        )
    return ParsedChord(
        root_pc=root_pc,
        pitch_classes=frozenset(pcs),
        raw=chord_str,  # keep original (with slash) for diagnostics
    )
