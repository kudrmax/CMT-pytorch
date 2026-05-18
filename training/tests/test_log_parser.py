"""Tests for training.log_parser."""
from __future__ import annotations

from pathlib import Path

from training.log_parser import parse_eval_metrics, EpochMetrics

# Realistic snippet matching authorial trainer.py log format (with logger prefixes
# and `accuracy_<name>/<mode>` keys, NOT `accuracy/<name>/<mode>`). Verified against
# real run.py output 2026-04-29.
SAMPLE_LOG = """\
I colab-training 04-29 11:47:38.350 trainer.py:121] ==========train 1 epoch==========
I colab-training 04-29 11:49:31.046 utils.py:43] nll/train: 23.81
I colab-training 04-29 11:49:31.048 utils.py:45] accuracy_rhythm/train: 0.4961
I colab-training 04-29 11:49:31.048 utils.py:45] accuracy_pitch/train: 0.0462
I colab-training 04-29 11:50:24.124 trainer.py:126] ==========valid 1 epoch==========
I colab-training 04-29 11:50:26.643 utils.py:43] nll/eval: 13.5105
I colab-training 04-29 11:50:26.643 utils.py:43] nll_pitch/eval: 12.4085
I colab-training 04-29 11:50:26.643 utils.py:43] nll_rhythm/eval: 1.1020
I colab-training 04-29 11:50:26.643 utils.py:45] accuracy_rhythm/eval: 0.6500
I colab-training 04-29 11:50:26.643 utils.py:45] accuracy_pitch/eval: 0.3000
I colab-training 04-29 11:50:26.644 trainer.py:121] ==========train 2 epoch==========
I colab-training 04-29 11:51:25.648 utils.py:45] accuracy_rhythm/train: 0.5166
I colab-training 04-29 11:52:26.000 trainer.py:126] ==========valid 2 epoch==========
I colab-training 04-29 11:52:30.000 utils.py:43] nll_pitch/eval: 0.16
I colab-training 04-29 11:52:30.000 utils.py:43] nll_rhythm/eval: 0.24
I colab-training 04-29 11:52:30.000 utils.py:45] accuracy_rhythm/eval: 0.7000
I colab-training 04-29 11:52:30.000 utils.py:45] accuracy_pitch/eval: 0.3500
"""


def test_parse_eval_metrics(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    log.write_text(SAMPLE_LOG)
    metrics = parse_eval_metrics(log)
    assert len(metrics) == 2
    m1 = metrics[0]
    assert isinstance(m1, EpochMetrics)
    assert m1.epoch == 1
    assert m1.rhythm_acc == 0.65
    assert m1.pitch_acc == 0.30
    assert m1.rhythm_loss == 1.1020
    assert m1.pitch_loss == 12.4085
    m2 = metrics[1]
    assert m2.epoch == 2
    assert m2.rhythm_acc == 0.70
    assert m2.pitch_acc == 0.35


def test_empty_log(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    log.write_text("")
    assert parse_eval_metrics(log) == []


def test_partial_log_no_eval_yet(tmp_path: Path) -> None:
    """Log with only training, no validation epoch yet."""
    log = tmp_path / "log.txt"
    log.write_text("==========train 1 epoch==========\n100 training steps\n")
    assert parse_eval_metrics(log) == []


from training.log_parser import best_epoch_by


def _make(*tuples: tuple[int, float, float]) -> list[EpochMetrics]:
    return [
        EpochMetrics(epoch=e, rhythm_acc=r, pitch_acc=p, rhythm_loss=0.0, pitch_loss=0.0)
        for e, r, p in tuples
    ]


def test_best_by_rhythm_acc_simple() -> None:
    metrics = _make((10, 0.5, 0.3), (20, 0.6, 0.35), (30, 0.55, 0.40))
    assert best_epoch_by(metrics, "rhythm_acc") == 20


def test_best_by_pitch_acc() -> None:
    metrics = _make((10, 0.5, 0.3), (20, 0.6, 0.35), (30, 0.55, 0.40))
    assert best_epoch_by(metrics, "pitch_acc") == 30


def test_only_at_multiples_of_10() -> None:
    """Skips epoch 15 (no checkpoint at that epoch by trainer.py convention)."""
    metrics = _make((10, 0.5, 0.0), (15, 0.99, 0.0), (20, 0.6, 0.0))
    assert best_epoch_by(metrics, "rhythm_acc", only_at_multiples_of=10) == 20


def test_ties_pick_smaller_epoch() -> None:
    """Tie-break: smaller epoch wins."""
    metrics = _make((20, 0.7, 0.0), (40, 0.7, 0.0), (60, 0.7, 0.0))
    assert best_epoch_by(metrics, "rhythm_acc") == 20


def test_empty_raises() -> None:
    import pytest
    with pytest.raises(ValueError, match="empty"):
        best_epoch_by([], "rhythm_acc")
