# CMT Training Orchestrator

Resume-aware Python orchestrator that trains CMT on jazz data using
authorial `run.py` (variant 2P-PV-RP from paper Choi et al. 2021).

## Quickstart

```bash
# 1. Prerequisites: dataset prepared (jazz/wjazzd/data/midi/ + 1-key pkl_files/)
#    See jazz/wjazzd/README.md

# 2. Install training-time deps
.venv/bin/pip install -r requirements-training.txt

# 3. Run orchestrator (idempotent — safe to re-run)
.venv/bin/python -m training.train --hparams hparams_jazz_16bars.yaml

# Output: result/paper/16bars/best_jazz_model_16bars.pth.tar
#         + summary.json + split_info.json + training_artefacts/{idx001,idx002}/
```

`--hparams` defaults to `hparams_jazz_8bars.yaml`; pass another
`hparams_jazz_{N}bars.yaml` for a different context length.

## What it does

1. **Preprocessing** — runs authorial `preprocess.py --shift` if 12-keys pkl absent.
2. **Phase 1** — trains Rhythm Decoder on 12-keys data. Output:
   `result/paper/{N}bars/training_artefacts/idx001/`.
3. **Best RD selection** — parses `idx001/log.txt`, finds epoch with
   max `accuracy_rhythm/eval` (only at multiples of 10 — checkpoint epochs).
4. **Phase 2** — trains Pitch Decoder on 1-key data, restoring RD weights from
   best Phase 1 epoch. Authorial `run.py --load_rhythm`. Output:
   `result/paper/{N}bars/training_artefacts/idx002/`.
5. **Best PD selection** — finds epoch with max `accuracy_pitch/eval`.
6. **Finalize** — copies best PD checkpoint to
   `result/paper/{N}bars/best_jazz_model_{N}bars.pth.tar`,
   writes `split_info.json` and `summary.json`. Warns if best PD epoch < 30
   (likely overfitting — anti-overfit recipe in spec).

## Resume

Idempotent. State detected from
`result/paper/{N}bars/training_artefacts/idxNNN/model/checkpoint_*.pth.tar`:
- Missing → start fresh
- Partial → `run.py --restore_epoch <last>` continues
- Complete → skip phase, proceed to next step

The baseline `hparams_jazz_{N}bars.yaml` is never modified by the
orchestrator. Per-phase derived YAMLs are written to
`training_artefacts/{idx001,idx002}/hparams.yaml`. Phase 2's derived
file gets `experiment.restore_rhythm.epoch` set to the best Phase 1
epoch automatically.

## How paths work

Each `hparams_jazz_{N}bars.yaml` is a **baseline configuration** —
pure reference, never executed directly by `run.py`. The orchestrator
derives per-phase files at runtime under `training_artefacts/`:

- `training_artefacts/idx001/hparams.yaml` — for Phase 1 (RD on 12-keys data)
- `training_artefacts/idx002/hparams.yaml` — for Phase 2 (PD with frozen RD on 1-key data)

Both derived files override:
- `asset_root` → `<--result-root>/training_artefacts`
- `data_io.path` → `<--data-root>/pkl_files/<canonical_pkl_subdir>`

Phase 2 derived also overrides:
- `experiment.restore_rhythm.epoch` → best Phase 1 epoch by validation rhythm accuracy

This keeps the baseline YAML pristine for git, while CLI args drive
all path injection.

## Authorial Code Patches

Single 1-line change to `run.py`: added `--hparams <path>` CLI argument
(default `hparams.yaml` for backward compat). Documented in `run.py` git
diff. No changes to `model.py`, `trainer.py`, `dataset.py`.

## CLI

```
python -m training.train [options]

Options:
  --data-root PATH        data dir (default: <repo>/jazz/wjazzd/data)
  --result-root PATH      result dir (default: <repo>/result)
  --hparams PATH          hparams YAML (default: <repo>/hparams_jazz_8bars.yaml)
  --gpu-index INT         GPU index for run.py (default: 0)
  --ngpu INT              Number of GPUs (default: 1)
  --max-epoch INT         Override experiment.max_epoch from hparams (for smoke testing)
```

## Tests

```bash
.venv/bin/python -m pytest tests/test_log_parser.py tests/test_resume_detector.py \
    tests/test_setup_hparams.py tests/test_train.py -v
```

20+ unit tests covering log parser, resume detection, hparams setup, and
subprocess command construction (mocked — no actual GPU training).

## Architecture

See `docs/superpowers/specs/2026-04-29-cmt-training-orchestrator-design.md`
for design rationale, anti-overfit strategy, and out-of-scope items.

## License

This module is part of the CMT-pytorch fork (`kudrmax/CMT-pytorch`) and
inherits its license.
