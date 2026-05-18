"""Unit tests for jazz.wjazzd.dataset_converter.chord_parser."""
from __future__ import annotations

import pytest

from jazz.wjazzd.dataset_converter import chord_parser as cp


class TestRoot:
    @pytest.mark.parametrize("chord_str,expected_root_pc", [
        ("C", 0),
        ("D", 2),
        ("E", 4),
        ("F", 5),
        ("G", 7),
        ("A", 9),
        ("B", 11),
        ("Bb", 10),
        ("Eb", 3),
        ("Ab", 8),
        ("Db", 1),
        ("Gb", 6),
        ("F#", 6),
        ("C#", 1),
        ("D#", 3),
    ])
    def test_root_pitch_class(self, chord_str: str, expected_root_pc: int) -> None:
        result = cp.parse_chord(chord_str)
        assert result.root_pc == expected_root_pc, (
            f"parse_chord({chord_str!r}).root_pc = {result.root_pc}, "
            f"expected {expected_root_pc}"
        )


class TestErrors:
    @pytest.mark.parametrize("bad_input", ["", "NC", "N.C.", "Q", "9C"])
    def test_invalid_raises(self, bad_input: str) -> None:
        with pytest.raises(cp.ChordParseError):
            cp.parse_chord(bad_input)


class TestTriadsAndSevenths:
    @pytest.mark.parametrize("chord_str,expected_pcs", [
        # Triads
        ("C",       {0, 4, 7}),
        ("Cm",      {0, 3, 7}),
        ("C-",      {0, 3, 7}),  # jazzomat: '-' = minor
        ("Cdim",    {0, 3, 6}),
        ("C°",      {0, 3, 6}),
        ("Caug",    {0, 4, 8}),
        ("C+",      {0, 4, 8}),
        # Sevenths
        ("C7",      {0, 4, 7, 10}),
        ("Cmaj7",   {0, 4, 7, 11}),
        ("Cj7",     {0, 4, 7, 11}),  # jazzomat: 'j' = maj
        ("Cm7",     {0, 3, 7, 10}),
        ("C-7",     {0, 3, 7, 10}),  # jazzomat
        ("Cm7b5",   {0, 3, 6, 10}),
        ("Cø",      {0, 3, 6, 10}),  # half-diminished
        ("Cdim7",   {0, 3, 6, 9}),
        ("C°7",     {0, 3, 6, 9}),
        ("Cmmaj7",  {0, 3, 7, 11}),  # m(maj7), rare
        # 6 chords
        ("C6",      {0, 4, 7, 9}),
        ("Cm6",     {0, 3, 7, 9}),
    ])
    def test_quality(self, chord_str: str, expected_pcs: set[int]) -> None:
        result = cp.parse_chord(chord_str)
        assert set(result.pitch_classes) == expected_pcs, (
            f"parse_chord({chord_str!r}).pitch_classes = "
            f"{sorted(result.pitch_classes)}, expected {sorted(expected_pcs)}"
        )


class TestAlterations:
    @pytest.mark.parametrize("chord_str,expected_pcs", [
        # Single alteration on dominant 7
        ("C7b9",   {0, 4, 7, 10, 1}),
        ("C7#9",   {0, 4, 7, 10, 3}),
        ("C7#11",  {0, 4, 7, 10, 6}),
        ("C7b13",  {0, 4, 7, 10, 8}),
        ("C7b5",   {0, 4, 6, 10}),     # b5 replaces 5
        ("C7#5",   {0, 4, 8, 10}),     # #5 replaces 5
        # Multiple alterations
        ("C7b9b13",  {0, 4, 7, 10, 1, 8}),
        ("C7#9#11",  {0, 4, 7, 10, 3, 6}),
        # alt = b5+#5+b9+#9 on dominant 7
        ("C7alt",  {0, 4, 6, 8, 10, 1, 3}),
        # Natural 9, 11, 13 — NOT added (Решение 7b)
        ("C9",     {0, 4, 7, 10}),
        ("C11",    {0, 4, 7, 10}),
        ("C13",    {0, 4, 7, 10}),
        # Mixed natural + altered: only altered are added
        ("C13b9",  {0, 4, 7, 10, 1}),
    ])
    def test_alterations(self, chord_str: str, expected_pcs: set[int]) -> None:
        result = cp.parse_chord(chord_str)
        assert set(result.pitch_classes) == expected_pcs


class TestSus:
    @pytest.mark.parametrize("chord_str,expected_pcs", [
        ("Csus4",   {0, 5, 7}),
        ("Csus2",   {0, 2, 7}),
        ("C7sus4",  {0, 5, 7, 10}),
        ("Csus",    {0, 5, 7}),       # bare 'sus' = sus4
    ])
    def test_sus(self, chord_str: str, expected_pcs: set[int]) -> None:
        result = cp.parse_chord(chord_str)
        assert set(result.pitch_classes) == expected_pcs


class TestSlash:
    @pytest.mark.parametrize("chord_str,expected_pcs", [
        # Slash should not affect pitch_classes (per Решение 13: bass comes from beats.bass_pitch)
        ("C/E",       {0, 4, 7}),
        ("Cm7/Bb",    {0, 3, 7, 10}),
        ("D7/F#",     {2, 6, 9, 0}),
        ("F/A",       {5, 9, 0}),
    ])
    def test_slash_pitch_classes(self, chord_str: str, expected_pcs: set[int]) -> None:
        result = cp.parse_chord(chord_str)
        assert set(result.pitch_classes) == expected_pcs


class TestWJazzDCorpusPatterns:
    """Patterns from WJazzD corpus not covered by unit tests above."""

    @pytest.mark.parametrize("chord_str,expected_pcs", [
        # 'o' as ASCII alternative for '°' (diminished)
        ("Co",    {0, 3, 6}),       # diminished triad
        ("Co7",   {0, 3, 6, 9}),    # diminished 7th
        ("Eo",    {4, 7, 10}),
        ("Eo7",   {4, 7, 10, 1}),
        ("Dbo",   {1, 4, 7}),
        ("Dbo7",  {1, 4, 7, 10}),
        # Jazzomat: '79b' = dominant 7 with b9 (alteration order non-standard)
        ("C79b",  {0, 4, 7, 10, 1}),   # C7b9
        ("D79b",  {2, 6, 9, 0, 3}),    # D7b9
        ("Bb79b", {10, 2, 5, 8, 11}),  # Bb7b9
        # Jazzomat: '79#' = dominant 7 with #9
        ("C79#",  {0, 4, 7, 10, 3}),   # C7#9
        ("A79#",  {9, 1, 4, 7, 0}),    # A7#9
        # Jazzomat: 'j7911#' = maj7 with natural 9, #11
        ("Ebj7911#", {3, 7, 10, 2, 9}),  # Ebmaj7 + #11 (6 semitones from root)
        ("Bj7911#",  {11, 3, 6, 10, 5}),
    ])
    def test_corpus_patterns(self, chord_str: str, expected_pcs: set[int]) -> None:
        result = cp.parse_chord(chord_str)
        assert set(result.pitch_classes) == expected_pcs, (
            f"parse_chord({chord_str!r}).pitch_classes = "
            f"{sorted(result.pitch_classes)}, expected {sorted(expected_pcs)}"
        )
