"""Per-epoch CSV logging for CMT trainer (pure I/O, no torch).

Writes one row per epoch to `{asset_path}/epochs.csv` with train + val
metrics side by side, mirroring the MINGUS-fork pattern
(`MINGUS/B_train/train.py::_append_epoch_row`) but with CMT's two-head
schema (rhythm + pitch). In Phase 1 (rhythm_only=True) pitch fields are
written as empty strings — distinguishes "head not trained" from
"trained to 0" for downstream pandas analysis.

Split out from trainer.py so the logic is unit-testable without torch.
"""
from __future__ import annotations

import csv
import os
from typing import Any


EPOCHS_CSV_FIELDNAMES: list[str] = [
    "epoch",
    "train_nll", "train_nll_pitch", "train_nll_rhythm",
    "eval_nll",  "eval_nll_pitch",  "eval_nll_rhythm",
    "train_accuracy_pitch", "train_accuracy_rhythm",
    "eval_accuracy_pitch",  "eval_accuracy_rhythm",
]


def build_epoch_row(
    mode: str,
    rhythm_only: bool,
    losses: dict[str, Any],
    results: dict[str, Any],
    existing_row: dict[str, Any],
) -> dict[str, Any]:
    """Merge one phase's metrics into the row buffer (immutable update).

    Args:
        mode: 'train' or 'eval' — selects which columns to populate.
        rhythm_only: True for Phase 1 (no pitch head trained) — pitch
            columns become "" so pandas reads them as NaN.
        losses: dict with keys 'nll/<mode>', 'nll_pitch/<mode>',
            'nll_rhythm/<mode>'. Produced by trainer._epoch() at line 205-207.
        results: dict with 'accuracy_rhythm/<mode>' and (unless rhythm_only)
            'accuracy_pitch/<mode>'.
        existing_row: previously built row (e.g. result of a prior call
            with the other mode); returned dict is a copy with new keys merged.

    Returns:
        New dict with existing_row's keys plus the mode-specific keys
        (`{mode}_nll`, `{mode}_nll_pitch`, `{mode}_nll_rhythm`,
        `{mode}_accuracy_pitch`, `{mode}_accuracy_rhythm`).
    """
    row = dict(existing_row)
    row[f"{mode}_nll"] = losses[f"nll/{mode}"]
    row[f"{mode}_nll_rhythm"] = losses[f"nll_rhythm/{mode}"]
    row[f"{mode}_accuracy_rhythm"] = results[f"accuracy_rhythm/{mode}"]
    if rhythm_only:
        row[f"{mode}_nll_pitch"] = ""
        row[f"{mode}_accuracy_pitch"] = ""
    else:
        row[f"{mode}_nll_pitch"] = losses[f"nll_pitch/{mode}"]
        row[f"{mode}_accuracy_pitch"] = results[f"accuracy_pitch/{mode}"]
    return row


def append_epoch_row(csv_path: str, row: dict[str, Any]) -> None:
    """Append one row to epochs.csv. Writes header if file doesn't exist.

    Missing keys default to "" (robustness against partial rows — same
    defensive pattern as BebopNet's _append_epoch_csv).
    """
    file_exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCHS_CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in EPOCHS_CSV_FIELDNAMES})


def truncate_epochs_csv(csv_path: str, keep_below_or_equal: int) -> None:
    """Drop rows whose epoch > keep_below_or_equal. No-op on edge cases.

    Called from trainer.train() before the epoch loop when restoring from
    a checkpoint — keeps the CSV's epoch history monotonic and dedup-free.
    keep_below_or_equal should be the `restore_epoch` value (already
    `checkpoint['epoch']` from load_model).

    Atomic rewrite via tmp file + os.replace so a crash mid-write leaves
    the original CSV intact.

    Edge cases (all no-op):
      - keep_below_or_equal < 0 (fresh start signaled by run.py default -1)
      - csv_path doesn't exist
      - csv_path is empty (0 bytes)
    """
    if keep_below_or_equal < 0:
        return
    if not os.path.isfile(csv_path):
        return
    if os.path.getsize(csv_path) == 0:
        return
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        kept = [r for r in reader if int(r["epoch"]) <= keep_below_or_equal]
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCHS_CSV_FIELDNAMES)
        writer.writeheader()
        for r in kept:
            writer.writerow({k: r.get(k, "") for k in EPOCHS_CSV_FIELDNAMES})
    os.replace(tmp_path, csv_path)
