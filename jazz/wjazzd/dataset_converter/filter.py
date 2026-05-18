"""Solo eligibility checks for WJazzD → CMT conversion.

Drops solos that are incompatible with CMT preprocess.py (4/4 only) or too short.
Empirical scope (per docs/cmt-jazz-training-brainstorm.md Решение 16):
  - 4/4 solos:           435 (95.4%) — kept
  - 3/4, 5/4, 6/8, etc.: 21  (4.6%)  — dropped
"""
from __future__ import annotations

from dataclasses import dataclass

MIN_BARS = 8  # CMT preprocess uses 8-bar windows; shorter solos yield 0 instances


@dataclass(frozen=True)
class SoloMeta:
    """Metadata for solo eligibility check."""

    melid: int
    signature: str             # solo_info.signature
    n_bars: int                # max(beats.bar) - min(beats.bar) + 1
    beat_signatures: set[str]  # distinct beats.signature values


def should_skip(meta: SoloMeta) -> tuple[bool, str]:
    """Return (skip, reason). reason is empty string if not skipping."""
    if meta.signature != "4/4":
        return True, f"signature {meta.signature!r} != 4/4"
    non_empty = {s for s in meta.beat_signatures if s}
    if len(non_empty) > 1:
        return True, f"internal signature change: {sorted(non_empty)}"
    if meta.n_bars < MIN_BARS:
        return True, f"only {meta.n_bars} bars, need >= {MIN_BARS}"
    return False, ""
