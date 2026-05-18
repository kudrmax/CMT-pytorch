"""Tests for training.epochs_csv (pure I/O, no torch)."""
from __future__ import annotations

import csv
from pathlib import Path

from training.epochs_csv import (
    EPOCHS_CSV_FIELDNAMES,
    append_epoch_row,
    build_epoch_row,
    truncate_epochs_csv,
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return (header, rows) for assertion convenience."""
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    return header, rows


def _full_phase2_row(epoch: int) -> dict[str, object]:
    """Return a row with all 11 fields populated (Phase 2-style)."""
    return {
        "epoch": epoch,
        "train_nll": 2.5, "train_nll_pitch": 2.0, "train_nll_rhythm": 0.5,
        "eval_nll":  3.0, "eval_nll_pitch":  2.4, "eval_nll_rhythm":  0.6,
        "train_accuracy_pitch": 0.30, "train_accuracy_rhythm": 0.70,
        "eval_accuracy_pitch":  0.25, "eval_accuracy_rhythm":  0.65,
    }


def test_clean_start_creates_header_and_appends(tmp_path: Path) -> None:
    csv_path = str(tmp_path / "epochs.csv")
    append_epoch_row(csv_path, _full_phase2_row(1))
    append_epoch_row(csv_path, _full_phase2_row(2))
    append_epoch_row(csv_path, _full_phase2_row(3))

    header, rows = _read_csv(Path(csv_path))
    assert header == EPOCHS_CSV_FIELDNAMES
    assert len(rows) == 3
    assert [int(r["epoch"]) for r in rows] == [1, 2, 3]
    assert rows[0]["train_nll"] == "2.5"
    assert rows[2]["eval_accuracy_rhythm"] == "0.65"


def test_resume_truncate_then_append_no_duplicates(tmp_path: Path) -> None:
    csv_path = str(tmp_path / "epochs.csv")
    for epoch in (1, 2, 3):
        append_epoch_row(csv_path, _full_phase2_row(epoch))

    truncate_epochs_csv(csv_path, keep_below_or_equal=2)

    header, rows = _read_csv(Path(csv_path))
    assert header == EPOCHS_CSV_FIELDNAMES
    assert [int(r["epoch"]) for r in rows] == [1, 2]

    append_epoch_row(csv_path, _full_phase2_row(3))
    _, rows = _read_csv(Path(csv_path))
    assert [int(r["epoch"]) for r in rows] == [1, 2, 3]


def test_truncate_edge_cases(tmp_path: Path) -> None:
    csv_path = str(tmp_path / "epochs.csv")

    truncate_epochs_csv(csv_path, keep_below_or_equal=5)
    assert not Path(csv_path).exists()

    for epoch in (1, 2):
        append_epoch_row(csv_path, _full_phase2_row(epoch))
    truncate_epochs_csv(csv_path, keep_below_or_equal=-1)
    _, rows = _read_csv(Path(csv_path))
    assert [int(r["epoch"]) for r in rows] == [1, 2]

    truncate_epochs_csv(csv_path, keep_below_or_equal=0)
    _, rows = _read_csv(Path(csv_path))
    assert rows == []


def test_rhythm_only_empty_pitch_roundtrip(tmp_path: Path) -> None:
    losses_train = {"nll/train": 1.2, "nll_pitch/train": 0.0, "nll_rhythm/train": 1.2}
    results_train = {"accuracy_rhythm/train": 0.7}

    row = build_epoch_row("train", rhythm_only=True,
                          losses=losses_train, results=results_train,
                          existing_row={})

    assert row["train_nll_pitch"] == ""
    assert row["train_accuracy_pitch"] == ""
    assert row["train_nll_rhythm"] == 1.2
    assert row["train_accuracy_rhythm"] == 0.7
    assert row["train_nll"] == 1.2

    row["epoch"] = 1
    csv_path = str(tmp_path / "epochs.csv")
    append_epoch_row(csv_path, row)

    _, rows = _read_csv(Path(csv_path))
    assert rows[0]["train_nll_pitch"] == ""
    assert rows[0]["train_accuracy_pitch"] == ""
    assert rows[0]["train_nll_rhythm"] == "1.2"
    assert rows[0]["eval_nll"] == ""


def test_two_phase_merge_preserves_train_fields(tmp_path: Path) -> None:
    losses_train = {"nll/train": 1.5, "nll_pitch/train": 1.0, "nll_rhythm/train": 0.5}
    results_train = {"accuracy_rhythm/train": 0.6, "accuracy_pitch/train": 0.2}
    losses_eval = {"nll/eval": 1.8, "nll_pitch/eval": 1.3, "nll_rhythm/eval": 0.5}
    results_eval = {"accuracy_rhythm/eval": 0.55, "accuracy_pitch/eval": 0.18}

    row = build_epoch_row("train", rhythm_only=False,
                          losses=losses_train, results=results_train,
                          existing_row={})
    row = build_epoch_row("eval", rhythm_only=False,
                          losses=losses_eval, results=results_eval,
                          existing_row=row)

    assert row["train_nll"] == 1.5
    assert row["train_accuracy_pitch"] == 0.2
    assert row["eval_nll"] == 1.8
    assert row["eval_accuracy_pitch"] == 0.18
    assert row["eval_accuracy_rhythm"] == 0.55


def test_full_phase2_row_no_empty_fields(tmp_path: Path) -> None:
    losses_train = {"nll/train": 2.5, "nll_pitch/train": 2.0, "nll_rhythm/train": 0.5}
    results_train = {"accuracy_rhythm/train": 0.7, "accuracy_pitch/train": 0.3}
    losses_eval = {"nll/eval": 3.0, "nll_pitch/eval": 2.4, "nll_rhythm/eval": 0.6}
    results_eval = {"accuracy_rhythm/eval": 0.65, "accuracy_pitch/eval": 0.25}

    row = build_epoch_row("train", rhythm_only=False,
                          losses=losses_train, results=results_train,
                          existing_row={})
    row = build_epoch_row("eval", rhythm_only=False,
                          losses=losses_eval, results=results_eval,
                          existing_row=row)
    row["epoch"] = 7

    csv_path = str(tmp_path / "epochs.csv")
    append_epoch_row(csv_path, row)

    _, rows = _read_csv(Path(csv_path))
    assert len(rows) == 1
    for field in EPOCHS_CSV_FIELDNAMES:
        assert rows[0][field] != "", f"unexpected empty field: {field}"
