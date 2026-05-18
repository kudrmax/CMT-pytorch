"""Parse trainer.py log.txt → per-epoch validation metrics.

Used by orchestrator to find best-by-rhythm-accuracy (Phase 1) and
best-by-pitch-accuracy (Phase 2) for paper-faithful checkpoint selection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EpochMetrics:
    """Validation metrics for one epoch (eval mode)."""
    epoch: int
    rhythm_acc: float
    pitch_acc: float       # accuracy/onset_only/eval (excludes hold/rest tokens)
    rhythm_loss: float
    pitch_loss: float


_EPOCH_HEADER_RE = re.compile(r"==========valid\s+(\d+)\s+epoch==========")
# cal_metrics in utils/metrics.py emits keys as "<metric>_<name>/<mode>",
# e.g. "accuracy_rhythm/eval", "accuracy_pitch/eval". The pitch one is
# computed by accuracy_pitch() which counts onset tokens only — that's our
# pitch_acc field (paper's "accuracy onset only").
_RHYTHM_ACC_RE  = re.compile(r"accuracy_rhythm/eval:\s*([\d.]+)")
_PITCH_ACC_RE   = re.compile(r"accuracy_pitch/eval:\s*([\d.]+)")
_RHYTHM_LOSS_RE = re.compile(r"nll_rhythm/eval:\s*([\d.]+)")
_PITCH_LOSS_RE  = re.compile(r"nll_pitch/eval:\s*([\d.]+)")


def parse_eval_metrics(log_path: Path) -> list[EpochMetrics]:
    """Parse trainer.py log.txt → list of EpochMetrics in epoch order.

    Skips epochs where any metric is missing (incomplete or corrupt log).
    """
    if not log_path.is_file():
        return []
    text = log_path.read_text()
    out: list[EpochMetrics] = []
    parts = re.split(_EPOCH_HEADER_RE, text)
    # parts = [header_before, epoch1_str, block1, epoch2_str, block2, ...]
    for i in range(1, len(parts), 2):
        epoch = int(parts[i])
        block = parts[i + 1] if i + 1 < len(parts) else ""
        ra = _RHYTHM_ACC_RE.search(block)
        pa = _PITCH_ACC_RE.search(block)
        rl = _RHYTHM_LOSS_RE.search(block)
        pl = _PITCH_LOSS_RE.search(block)
        if not (ra and pa and rl and pl):
            continue
        out.append(EpochMetrics(
            epoch=epoch,
            rhythm_acc=float(ra.group(1)),
            pitch_acc=float(pa.group(1)),
            rhythm_loss=float(rl.group(1)),
            pitch_loss=float(pl.group(1)),
        ))
    return out


def best_epoch_by(
    metrics: list[EpochMetrics],
    key: str,
    only_at_multiples_of: int = 10,
) -> int:
    """Return epoch with maximum value of `key` among saved checkpoints.

    Args:
        metrics: from parse_eval_metrics()
        key: attribute name on EpochMetrics — e.g. "rhythm_acc" or "pitch_acc"
        only_at_multiples_of: restrict to epochs that are multiples of this number
            (because trainer.py only saves checkpoints at multiples of 10).

    Ties resolve to smallest epoch.
    Raises ValueError on empty input or no eligible epochs.
    """
    if not metrics:
        raise ValueError("metrics is empty; cannot pick best epoch")
    eligible = [m for m in metrics if m.epoch % only_at_multiples_of == 0]
    if not eligible:
        raise ValueError(
            f"no epoch is a multiple of {only_at_multiples_of}; "
            f"available epochs: {[m.epoch for m in metrics]}"
        )
    best = max(eligible, key=lambda m: (getattr(m, key), -m.epoch))
    return best.epoch
