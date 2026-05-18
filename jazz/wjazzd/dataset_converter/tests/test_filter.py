"""Tests for jazz.wjazzd.dataset_converter.filter — solo eligibility checks."""
from __future__ import annotations

from jazz.wjazzd.dataset_converter.filter import SoloMeta, should_skip


def test_keeps_44_solo() -> None:
    meta = SoloMeta(melid=1, signature="4/4", n_bars=32, beat_signatures={"4/4"})
    skip, reason = should_skip(meta)
    assert skip is False
    assert reason == ""


def test_skips_34_solo() -> None:
    meta = SoloMeta(melid=2, signature="3/4", n_bars=32, beat_signatures={"3/4"})
    skip, reason = should_skip(meta)
    assert skip is True
    assert "signature" in reason


def test_skips_internal_signature_change() -> None:
    meta = SoloMeta(melid=3, signature="4/4", n_bars=32, beat_signatures={"4/4", "5/4"})
    skip, reason = should_skip(meta)
    assert skip is True
    assert "internal" in reason


def test_skips_short_solo() -> None:
    meta = SoloMeta(melid=4, signature="4/4", n_bars=4, beat_signatures={"4/4"})
    skip, reason = should_skip(meta)
    assert skip is True
    assert "bars" in reason


def test_keeps_minimum_8_bars() -> None:
    meta = SoloMeta(melid=5, signature="4/4", n_bars=8, beat_signatures={"4/4"})
    skip, _ = should_skip(meta)
    assert skip is False


def test_skips_empty_signature() -> None:
    meta = SoloMeta(melid=6, signature="", n_bars=32, beat_signatures={""})
    skip, reason = should_skip(meta)
    assert skip is True
