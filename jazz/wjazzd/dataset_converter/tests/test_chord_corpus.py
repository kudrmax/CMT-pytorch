"""Regression test: parser must handle ≥95% of unique chord strings in real wjazzd.db.

This test prevents silent degradation if parser logic regresses, and documents
which strings the parser currently doesn't handle (logged for analysis).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from jazz.wjazzd.dataset_converter.chord_parser import parse_chord, ChordParseError

# Threshold: at least this fraction of total chord OCCURRENCES (not unique strings)
# must parse. Weighted by frequency so we ensure the common case works.
PARSE_RATE_THRESHOLD = 0.95


def test_parse_rate_on_full_corpus(wjazzd_db_path: Path) -> None:
    """Parse every distinct chord in beats table; assert pass-rate >= threshold."""
    conn = sqlite3.connect(wjazzd_db_path)
    try:
        cursor = conn.execute("""
            SELECT DISTINCT chord, COUNT(*) AS cnt
            FROM beats
            WHERE chord IS NOT NULL AND chord != '' AND chord != 'NC'
            GROUP BY chord
            ORDER BY cnt DESC
        """)
        rows = cursor.fetchall()
    finally:
        conn.close()

    total_unique = len(rows)
    total_occurrences = sum(c for _, c in rows)
    failed: list[tuple[str, int, str]] = []
    for chord_str, count in rows:
        try:
            parse_chord(chord_str)
        except ChordParseError as e:
            failed.append((chord_str, count, str(e)))

    parsed_unique = total_unique - len(failed)
    parsed_occurrences = total_occurrences - sum(c for _, c, _ in failed)

    print(f"\nUnique chord strings in WJazzD: {total_unique}")
    print(f"Total chord occurrences (non-empty, non-NC): {total_occurrences}")
    print(f"Successfully parsed: {parsed_unique} unique ({parsed_occurrences} occurrences)")
    print(f"Failed to parse: {len(failed)} unique ({sum(c for _, c, _ in failed)} occurrences)")
    if failed:
        print("\nFailed chords (top 30 by frequency):")
        for chord, count, err in failed[:30]:
            print(f"  {count:6d}  {chord!r:30s}  {err}")

    rate = parsed_occurrences / total_occurrences if total_occurrences else 0
    assert rate >= PARSE_RATE_THRESHOLD, (
        f"Parse rate {rate:.3f} below threshold {PARSE_RATE_THRESHOLD}"
    )
