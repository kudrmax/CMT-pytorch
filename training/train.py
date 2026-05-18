"""Resume-aware orchestrator for CMT two-phase training.

Spawns authorial run.py + preprocess.py via subprocess. Uses our
training/resume_detector.py + log_parser.py + setup_hparams.py for state management.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from training.resume_detector import detect_state
from training.log_parser import parse_eval_metrics, best_epoch_by
from training.setup_hparams import derive_phase_hparams, load_hparams
from training.pkl_paths import pkl_dir_name, pkl_preprocess_params

REPO_ROOT = Path(__file__).resolve().parent.parent

TRAINING_ARTEFACTS_SUBDIR = "training_artefacts"


def _unbuffered_env() -> dict[str, str]:
    """Env with PYTHONUNBUFFERED=1 so child Python prints stream live to Colab."""
    import os
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_preprocess_shift(
    data_root: Path,
    *,
    num_bars: int,
    stride_bars: int,
    frame_per_bar: int,
    pitch_range: int,
    pkl_subdir: str,
) -> None:
    """Run authorial preprocess.py --shift (12-keys) with given preprocess params.

    pkl_subdir is the expected output folder under data_root/pkl_files/ — used
    only to write the `.done` marker on success. Caller computes it via
    training.pkl_paths.pkl_dir_name to keep names consistent.
    """
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    cmd = [
        sys.executable, str(REPO_ROOT / "preprocess.py"),
        "--root_dir", str(data_root),
        "--midi_dir", "midi",
        "--num_bars", str(num_bars),
        "--stride_bars", str(stride_bars),
        "--frame_per_bar", str(frame_per_bar),
        "--pitch_range", str(pitch_range),
        "--shift",
    ]
    print(f"==> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=_unbuffered_env())
    pkl_root = data_root / "pkl_files" / pkl_subdir
    if pkl_root.is_dir():
        (pkl_root / ".done").touch()


def run_phase_1(
    hparams_path: Path,
    restore_epoch: int | None,
    gpu_index: int = 0,
    ngpu: int = 1,
) -> None:
    """Run authorial run.py for Phase 1 (Rhythm Decoder training).

    asset_root and data_io.path are injected into hparams_path via
    derive_phase_hparams — run.py reads them from hparams, not from CLI.
    """
    cmd = [
        sys.executable, str(REPO_ROOT / "run.py"),
        "--idx", "1",
        "--gpu_index", str(gpu_index),
        "--ngpu", str(ngpu),
        "--hparams", str(hparams_path),
    ]
    if restore_epoch is not None:
        cmd.extend(["--restore_epoch", str(restore_epoch)])
    print(f"==> Running Phase 1: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=_unbuffered_env())


def run_phase_2(
    hparams_path: Path,
    restore_epoch: int | None,
    gpu_index: int = 0,
    ngpu: int = 1,
) -> None:
    """Run authorial run.py for Phase 2 (Pitch Decoder training).

    Includes --load_rhythm flag — authorial run.py uses it to restore RD
    weights from experiment.restore_rhythm.{idx,epoch} in hparams.
    asset_root and data_io.path are injected into hparams_path via
    derive_phase_hparams — run.py reads them from hparams, not from CLI.
    """
    cmd = [
        sys.executable, str(REPO_ROOT / "run.py"),
        "--idx", "2",
        "--gpu_index", str(gpu_index),
        "--ngpu", str(ngpu),
        "--load_rhythm",
        "--hparams", str(hparams_path),
    ]
    if restore_epoch is not None:
        cmd.extend(["--restore_epoch", str(restore_epoch)])
    print(f"==> Running Phase 2: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=_unbuffered_env())


OVERFIT_EPOCH_THRESHOLD = 30  # warn if best PD epoch < this


def _list_solos(split_dir: Path) -> list[str]:
    """Return sorted list of solo (directory) names in a split."""
    if not split_dir.is_dir():
        return []
    return sorted(d.name for d in split_dir.iterdir() if d.is_dir())


def _count_pkl(split_dir: Path) -> int:
    if not split_dir.is_dir():
        return 0
    return sum(1 for _ in split_dir.rglob("*.pkl"))


def finalize(
    result_root: Path,
    data_root: Path,
    best_pd_epoch: int,
    best_rd_epoch: int,
    max_epoch: int,
    best_rhythm_acc: float,
    best_pitch_acc: float,
    pkl_subdir_1key: str,
    final_model_name: str,
    hparams_path: Path,
) -> None:
    """Copy best Phase 2 checkpoint, write split_info.json + summary.json."""
    src = result_root / TRAINING_ARTEFACTS_SUBDIR / "idx002" / "model" / f"checkpoint_{best_pd_epoch}.pth.tar"
    if not src.is_file():
        raise FileNotFoundError(f"best PD checkpoint not found: {src}")
    dst = result_root / final_model_name
    shutil.copy2(src, dst)

    pkl_root_1key = data_root / "pkl_files" / pkl_subdir_1key
    splits = {s: _list_solos(pkl_root_1key / s) for s in ("train", "eval", "test")}
    counts = {s: _count_pkl(pkl_root_1key / s) for s in ("train", "eval", "test")}
    split_info = {
        "preprocess_seed": 0,
        "data_ratio": [0.8, 0.1, 0.1],
        "instances": counts,
        "solos": splits,
    }
    (result_root / "split_info.json").write_text(json.dumps(split_info, indent=2))

    overfit = best_pd_epoch < OVERFIT_EPOCH_THRESHOLD
    summary = {
        "phase1_best_epoch": best_rd_epoch,
        "phase1_best_rhythm_acc": best_rhythm_acc,
        "phase2_best_epoch": best_pd_epoch,
        "phase2_best_pitch_acc_onset_only": best_pitch_acc,
        "max_epoch": max_epoch,
        "overfit_warning": overfit,
        "hparams_used": hparams_path.name,
    }
    (result_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"==> Final artifacts:")
    print(f"  Model: {dst}")
    print(f"  Split: {result_root / 'split_info.json'}")
    print(f"  Summary: {result_root / 'summary.json'}")
    if overfit:
        print(f"  WARN: best Phase 2 epoch = {best_pd_epoch} (< {OVERFIT_EPOCH_THRESHOLD}) — likely overfitting.")
        print(f"        Consider: dropout 0.2 → 0.3, weight_decay 0 → 0.0001 in {hparams_path.name}")


def main(argv: list[str] | None = None) -> int:
    """Resume-aware orchestrator for CMT two-phase training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "jazz" / "wjazzd" / "data",
        help="Path to data directory (contains wjazzd.db, midi/, pkl_files/)",
    )
    parser.add_argument(
        "--result-root", type=Path, default=REPO_ROOT / "result",
        help="Path where training results (training_artefacts/idx001/, training_artefacts/idx002/, best_jazz_model_{N}bars.pth.tar) are written",
    )
    parser.add_argument(
        "--hparams", type=Path, default=REPO_ROOT / "hparams_jazz_8bars.yaml",
        help="Path to baseline hparams YAML (will be derived per-phase, never modified)",
    )
    parser.add_argument(
        "--gpu-index", type=int, default=0,
        help="GPU index for run.py",
    )
    parser.add_argument(
        "--ngpu", type=int, default=1,
        help="Number of GPUs for run.py",
    )
    parser.add_argument(
        "--max-epoch", type=int, default=None,
        help="Override experiment.max_epoch from hparams (for smoke testing)",
    )
    args = parser.parse_args(argv)

    if not args.hparams.is_file():
        print(f"ERROR: hparams not found: {args.hparams}", file=sys.stderr)
        return 1

    hparams_data = load_hparams(args.hparams)
    max_epoch = args.max_epoch if args.max_epoch is not None else int(hparams_data["experiment"]["max_epoch"])

    num_bars, stride_bars, fpb, pitch_range = pkl_preprocess_params(hparams_data)
    pkl_subdir_1key = pkl_dir_name(num_bars=num_bars, stride_bars=stride_bars,
                                    frame_per_bar=fpb, pitch_range=pitch_range, shift=False)
    pkl_subdir_12keys = pkl_dir_name(num_bars=num_bars, stride_bars=stride_bars,
                                      frame_per_bar=fpb, pitch_range=pitch_range, shift=True)
    final_model_name = f"best_jazz_model_{num_bars}bars.pth.tar"

    # ===== Step 1: ensure 12-keys preprocessing =====
    state = detect_state(args.data_root, args.result_root, max_epoch, pkl_subdir_12keys)
    if state.needs_12keys_preproc:
        print("==> 12-keys preprocessing not found, running preprocess.py --shift ...")
        run_preprocess_shift(
            args.data_root,
            num_bars=num_bars, stride_bars=stride_bars,
            frame_per_bar=fpb, pitch_range=pitch_range,
            pkl_subdir=pkl_subdir_12keys,
        )

    # ===== Step 2: derive Phase 1 hparams and run Phase 1 =====
    artefacts_root = args.result_root / TRAINING_ARTEFACTS_SUBDIR
    phase1_hparams = artefacts_root / "idx001" / "hparams.yaml"
    derive_phase_hparams(
        args.hparams,
        phase1_hparams,
        asset_root=artefacts_root,
        data_path=args.data_root / "pkl_files" / pkl_subdir_12keys,
        max_epoch_override=args.max_epoch,
    )

    state = detect_state(args.data_root, args.result_root, max_epoch, pkl_subdir_12keys)
    if state.phase1.status != "complete":
        restore = state.phase1.last_epoch if state.phase1.status == "partial" else None
        run_phase_1(
            hparams_path=phase1_hparams,
            restore_epoch=restore,
            gpu_index=args.gpu_index,
            ngpu=args.ngpu,
        )

    # ===== Step 3: select best RD epoch =====
    log_p1 = artefacts_root / "idx001" / "log.txt"
    metrics_p1 = parse_eval_metrics(log_p1)
    if not metrics_p1:
        print(f"ERROR: no metrics in {log_p1}", file=sys.stderr)
        return 2
    try:
        best_rd = best_epoch_by(metrics_p1, "rhythm_acc")
    except ValueError as e:
        print(f"ERROR: cannot select best Phase 1 epoch: {e}", file=sys.stderr)
        return 4
    print(f"==> Best Phase 1 RD epoch: {best_rd} (rhythm_acc max)")

    # ===== Step 4: derive Phase 2 hparams =====
    phase2_hparams = artefacts_root / "idx002" / "hparams.yaml"
    derive_phase_hparams(
        args.hparams,
        phase2_hparams,
        asset_root=artefacts_root,
        data_path=args.data_root / "pkl_files" / pkl_subdir_1key,
        restore_rhythm_epoch=best_rd,
        max_epoch_override=args.max_epoch,
    )

    # ===== Step 5: Phase 2 =====
    state = detect_state(args.data_root, args.result_root, max_epoch, pkl_subdir_12keys)
    if state.phase2.status != "complete":
        restore = state.phase2.last_epoch if state.phase2.status == "partial" else None
        run_phase_2(
            hparams_path=phase2_hparams,
            restore_epoch=restore,
            gpu_index=args.gpu_index,
            ngpu=args.ngpu,
        )

    # ===== Step 6: select best PD epoch + finalize =====
    log_p2 = artefacts_root / "idx002" / "log.txt"
    metrics_p2 = parse_eval_metrics(log_p2)
    if not metrics_p2:
        print(f"ERROR: no metrics in {log_p2}", file=sys.stderr)
        return 3
    try:
        best_pd = best_epoch_by(metrics_p2, "pitch_acc")
    except ValueError as e:
        print(f"ERROR: cannot select best Phase 2 epoch: {e}", file=sys.stderr)
        return 5
    print(f"==> Best Phase 2 PD epoch: {best_pd} (pitch_acc max)")

    best_rd_metric = next(m for m in metrics_p1 if m.epoch == best_rd)
    best_pd_metric = next(m for m in metrics_p2 if m.epoch == best_pd)
    finalize(
        result_root=args.result_root,
        data_root=args.data_root,
        best_pd_epoch=best_pd,
        best_rd_epoch=best_rd,
        max_epoch=max_epoch,
        best_rhythm_acc=best_rd_metric.rhythm_acc,
        best_pitch_acc=best_pd_metric.pitch_acc,
        pkl_subdir_1key=pkl_subdir_1key,
        final_model_name=final_model_name,
        hparams_path=args.hparams,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
