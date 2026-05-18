"""Generate melody on full test set using trained CMT checkpoint.

Iterates pkl instances in --test-dir, primes the model with first --num-prime
tokens of each, samples remaining via model.sampling(), writes generated MIDI
to --out-dir.

Usage example (after `hf download` of the 16-bar paper checkpoint):
    python generate_test.py \\
        --checkpoint result/paper/16bars/best_jazz_model_16bars.pth.tar \\
        --hparams result/paper/16bars/hparams.yaml \\
        --test-dir jazz/wjazzd/data/pkl_files/instance_pkl_16bars_str4_fpb16_48p_ckey/test \\
        --out-dir result/paper/16bars/test_inference
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Make authorial code importable when running from any cwd
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from model import ChordConditionedMelodyTransformer  # noqa: E402
from utils.utils import pitch_to_midi  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True,
                        help="Directory containing test pkl files (recursive).")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-prime", type=int, default=16)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    with open(args.hparams) as f:
        hparams = yaml.safe_load(f)
    model_cfg = hparams["model"]
    frame_per_bar = int(model_cfg["frame_per_bar"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ChordConditionedMelodyTransformer(**model_cfg).to(device)
    ckpt = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch', '?')}")

    test_files = sorted(Path(args.test_dir).rglob("*.pkl"))
    if not test_files:
        print(f"ERROR: no pkl files in {args.test_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(test_files)} test instances")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, pkl_path in enumerate(test_files):
        with open(pkl_path, "rb") as f:
            inst = pickle.load(f)

        pitch_arr = np.asarray(inst["pitch"], dtype=np.int64)
        rhythm_arr = np.asarray(inst["rhythm"], dtype=np.int64)
        chord_dense = inst["chord"].toarray() if hasattr(inst["chord"], "toarray") else np.asarray(inst["chord"])
        chord_arr = np.asarray(chord_dense, dtype=np.float32)

        pitch_t = torch.from_numpy(pitch_arr).unsqueeze(0).to(device)        # (1, T)
        rhythm_t = torch.from_numpy(rhythm_arr).unsqueeze(0).to(device)      # (1, T)
        chord_t = torch.from_numpy(chord_arr).unsqueeze(0).to(device)        # (1, 12, T)

        prime_pitch = pitch_t[:, :args.num_prime]
        prime_rhythm = rhythm_t[:, :args.num_prime]

        with torch.no_grad():
            result = model.sampling(prime_rhythm, prime_pitch, chord_t, args.topk)

        gen_pitch = result["pitch"][0].cpu().numpy()
        gen_rhythm = result["rhythm"][0].cpu().numpy()
        chord_np = chord_t[0].cpu().numpy()

        out_path = args.out_dir / f"{pkl_path.stem}.mid"
        # Save generated melody alongside chord track for downstream listening + metrics
        pitch_to_midi(gen_pitch, chord_np[:, :-1], frame_per_bar, str(out_path))

        # Also save raw arrays in pkl for downstream metric computation
        out_pkl = args.out_dir / f"{pkl_path.stem}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump({
                "pitch": gen_pitch,
                "rhythm": gen_rhythm,
                "chord": inst["chord"],  # keep sparse representation
                "groundtruth_pitch": pitch_arr,
                "groundtruth_rhythm": rhythm_arr,
            }, f)

        if (i + 1) % 20 == 0 or (i + 1) == len(test_files):
            print(f"  {i + 1}/{len(test_files)}")

    print(f"Done. Output: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
