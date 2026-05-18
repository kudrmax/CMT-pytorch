"""Evaluate the trained best CMT checkpoint on the canonical test=40 set.

CMT's authorial pipeline never computes loss/accuracy on the test bucket
(test_loader is only used by trainer._sampling for MIDI generation). For
cross-model comparison with MINGUS and BebopNet, we need numeric metrics
on the same 40 files (split.json[test]).

This script runs ONCE after training:
  1. Loads `best_jazz_model_{N}bars.pth.tar` from `--result-root`.
  2. Reads split.json[test] (40 files) from `--split-json`.
  3. Builds a DataLoader from `pkl_files/<pkl_subdir>/test/<song>/*.pkl`
     filtered to only the 40 song_ids in the SSoT test bucket.
  4. Runs the same forward + cal_metrics loop as trainer._epoch('eval'),
     no backward, no LR step.
  5. Appends ``final_test_*`` fields to ``<result-root>/summary.json``.

Output keys (numbers averaged across batches, same convention as
trainer._epoch logs):
  final_test_loss            — total NLL (rhythm_loss + pitch_loss)
  final_test_ppl             — exp(final_test_loss)
  final_test_nll_pitch       — pitch_loss only
  final_test_nll_rhythm      — rhythm_loss only
  final_test_rhythm_acc      — accuracy_rhythm
  final_test_pitch_acc       — accuracy_pitch (onset-only, ignore rest/sustain)
  final_test_pitch_acc_full  — accuracy_pitch including hold/rest tokens
                               (matches paper Table 1 "Pitch including hold & rest")
  final_test_n_files         — actual file count after the split.json filter
  final_test_n_windows       — total instance windows evaluated
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset import collate_fn
from loss import FocalLoss
from model import ChordConditionedMelodyTransformer as CMT
from training.pkl_paths import pkl_dir_name
from utils.hparams import HParams
from utils.metrics import cal_metrics


class FilteredTestDataset(Dataset):
    """test-bucket pkl restricted to song_ids present in split.json[test]."""

    def __init__(self, pkl_test_root: Path, allowed_song_ids: set[str]):
        all_paths = sorted(glob.glob(os.path.join(str(pkl_test_root), "*/*.pkl")))
        self.file_paths = [p for p in all_paths if os.path.basename(os.path.dirname(p)) in allowed_song_ids]
        self.song_ids_present = sorted({os.path.basename(os.path.dirname(p)) for p in self.file_paths})

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], "rb") as f:
            instance = pickle.load(f)
        instance["chord"] = instance["chord"].toarray()
        return instance


def _evaluate(model, loader, criterion, metrics, device) -> tuple[dict, int]:
    """Mirror of trainer._epoch('eval') logic — no backward, no scheduler."""
    rhythm_criterion, pitch_criterion = criterion
    model.eval()

    from collections import defaultdict
    results = defaultdict(float)
    total_pitch_loss = 0.0
    total_rhythm_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for data in loader:
            for k in data.keys():
                data[k] = data[k].to(device)
            out = model(data["rhythm"], data["pitch"][:, :-1], data["chord"], False, False)
            rhythm_out = out["rhythm"].view(-1, out["rhythm"].size(-1))
            pitch_out = out["pitch"].view(-1, out["pitch"].size(-1))

            rhythm_loss = rhythm_criterion(rhythm_out, data["rhythm"][:, 1:].contiguous().view(-1))
            pitch_loss = pitch_criterion(pitch_out, data["pitch"][:, 1:].contiguous().view(-1))
            total_rhythm_loss += rhythm_loss.item()
            total_pitch_loss += pitch_loss.item()

            r = {}
            r.update(cal_metrics(rhythm_out, data["rhythm"][:, 1:].contiguous().view(-1),
                                 metrics, mode="eval", name="rhythm"))
            r.update(cal_metrics(pitch_out, data["pitch"][:, 1:].contiguous().view(-1),
                                 metrics, mode="eval", name="pitch"))
            # Additional pitch accuracy including hold/rest tokens (paper Table 1
            # "Pitch including hold & rest"). cal_metrics with name="pitch" maps
            # accuracy → accuracy_pitch (onset-only filter), so compute the full
            # accuracy here directly.
            pitch_target = data["pitch"][:, 1:].contiguous().view(-1)
            pitch_pred = pitch_out.argmax(dim=1)
            r["accuracy_pitch_full/eval"] = (pitch_pred == pitch_target).float().mean().item()
            for k, v in r.items():
                results[k] += v
            n_batches += 1

    if n_batches == 0:
        raise RuntimeError("No batches evaluated — DataLoader produced 0 batches "
                           "(likely batch_size > number of windows).")

    results = {k: v / n_batches for k, v in results.items()}
    avg_rhythm_loss = total_rhythm_loss / n_batches
    avg_pitch_loss = total_pitch_loss / n_batches
    avg_total_loss = avg_rhythm_loss + avg_pitch_loss
    return {
        "loss": avg_total_loss,
        "nll_pitch": avg_pitch_loss,
        "nll_rhythm": avg_rhythm_loss,
        "metrics": dict(results),
    }, n_batches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result-root", type=Path, required=True,
                        help="Run dir containing best_jazz_model_{N}bars.pth.tar + summary.json")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "jazz" / "wjazzd" / "data",
                        help="Path to data dir (must contain pkl_files/<pkl_subdir>/test/...)")
    parser.add_argument("--hparams", type=Path, required=True,
                        help="Path to hparams YAML used for training")
    parser.add_argument("--split-json", type=Path, required=True,
                        help="Path to diploma2 wjazzd_split.json")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: hparams.data_io.loader.batch_size)")
    args = parser.parse_args(argv)

    if not args.hparams.is_file():
        raise FileNotFoundError(f"hparams not found: {args.hparams}")
    if not args.split_json.is_file():
        raise FileNotFoundError(f"split-json not found: {args.split_json}")
    summary_path = args.result_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    config = HParams.load(str(args.hparams))
    prep = config.data_io["preprocess"]
    num_bars = int(prep["num_bars"])
    stride_bars = int(prep.get("stride_bars", num_bars // 2))
    fpb = int(prep["frame_per_bar"])
    pitch_range = int(prep["pitch_range"])
    pkl_subdir = pkl_dir_name(num_bars=num_bars, stride_bars=stride_bars,
                              frame_per_bar=fpb, pitch_range=pitch_range, shift=False)

    ckpt_path = args.result_root / f"best_jazz_model_{num_bars}bars.pth.tar"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"best checkpoint not found: {ckpt_path}")

    test_root = args.data_root / "pkl_files" / pkl_subdir / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"test pkl dir not found: {test_root}")

    with open(args.split_json) as f:
        split = json.load(f)
    canonical_test_ids = set(split["test"])
    print(f"==> split.json[test] has {len(canonical_test_ids)} canonical song_ids", flush=True)

    dataset = FilteredTestDataset(test_root, canonical_test_ids)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No test pkl matched split.json[test] under {test_root}. "
            f"Existing dirs: {sorted(p.name for p in test_root.iterdir())[:5]}..."
        )
    print(f"==> filtered test dataset: {len(dataset)} windows from "
          f"{len(dataset.song_ids_present)} files", flush=True)

    device = torch.device(
        f"cuda:{args.gpu_index}"
        if torch.cuda.is_available() and not args.no_cuda
        else "cpu"
    )
    batch_size = args.batch_size or int(config.data_io["loader"]["batch_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=False, collate_fn=collate_fn)

    model = CMT(**config.model).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"==> loaded {ckpt_path.name}", flush=True)

    criterion = (nn.NLLLoss().to(device), FocalLoss(gamma=2).to(device))
    metrics = config.experiment["metrics"]
    eval_result, n_batches = _evaluate(model, loader, criterion, metrics, device)

    accuracy_rhythm = eval_result["metrics"].get("accuracy_rhythm/eval")
    accuracy_pitch = eval_result["metrics"].get("accuracy_pitch/eval")
    accuracy_pitch_full = eval_result["metrics"].get("accuracy_pitch_full/eval")

    summary = json.loads(summary_path.read_text())
    summary["final_test_loss"] = float(eval_result["loss"])
    summary["final_test_ppl"] = float(math.exp(eval_result["loss"]))
    summary["final_test_nll_pitch"] = float(eval_result["nll_pitch"])
    summary["final_test_nll_rhythm"] = float(eval_result["nll_rhythm"])
    summary["final_test_rhythm_acc"] = float(accuracy_rhythm) if accuracy_rhythm is not None else None
    summary["final_test_pitch_acc"] = float(accuracy_pitch) if accuracy_pitch is not None else None
    summary["final_test_pitch_acc_full"] = float(accuracy_pitch_full) if accuracy_pitch_full is not None else None
    summary["final_test_n_files"] = len(dataset.song_ids_present)
    summary["final_test_n_windows"] = len(dataset)
    summary["final_test_note"] = (
        "evaluated on canonical test=40 (split.json[test]); held out from "
        "both gradient training and best-checkpoint selection. "
        "Metrics averaged across batches, matching trainer._epoch('eval') convention."
    )
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"==> final_test_ppl={summary['final_test_ppl']:.3f} | "
          f"rhythm_acc={summary['final_test_rhythm_acc']} | "
          f"pitch_acc={summary['final_test_pitch_acc']}", flush=True)
    print(f"==> wrote final_test_* to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
