# Running CMT Jazz Training on Google Colab Pro+

This document explains how to run CMT training on Colab Pro+ A100 using
the `training/colab_train.ipynb` notebook in this repository.

## Prerequisites

- Google Colab **Pro+** subscription (Pro is not enough — we rely on Pro+
  background execution to keep the runtime alive after closing the tab).
- Google Drive with at least 5 GB free space (training writes ~3.8 GB
  to `/content/drive/MyDrive/cmt-training/`).

## One-time setup

1. Open the notebook URL in a browser:
   `https://colab.research.google.com/github/kudrmax/CMT-pytorch/blob/master/training/colab_train.ipynb`

2. Configure runtime:
   - Menu: `Runtime` → `Change runtime type`
   - Hardware accelerator: **A100 GPU** (subject to availability — may fall
     back to V100 / L4 / T4 if A100 occupied)
   - Click `Save`

   **No background-execution checkbox.** Per [official Colab Pro+ docs](https://colab.research.google.com/notebooks/pro.ipynb):
   *"Colab Pro+ users have access to background execution, where notebooks
   will continue executing even after you've closed a browser tab. This is
   always enabled in Pro+ runtimes as long as you have compute units available."*
   The standalone checkbox was removed in 2024–2025; the feature is now always-on
   for Pro+ subscribers automatically.

3. (First-run-only) Approve Google Drive access:
   - When the cell hits `drive.mount(...)`, a dialog appears
   - Click `Connect to Google Drive` → choose your Google account → `Allow`

## Choosing config (8-bar vs 16-bar)

The notebook has a `HPARAMS` variable in the first cell:

```python
HPARAMS = "hparams_jazz_8bars.yaml"        # default — proven, paper-style
# HPARAMS = "hparams_jazz_16bars.yaml"     # 2× context, batch_size 16
```

Both configs are independent:
- Different result-root on Drive: `result_8bars/` vs `result_16bars/`
- Different pkl folders: `instance_pkl_8bars_str4_fpb16_48p_*` vs `instance_pkl_16bars_str4_fpb16_48p_*`
- Different final checkpoint: `best_jazz_model_8bars.pth.tar` vs `best_jazz_model_16bars.pth.tar`

To switch, comment/uncomment the line and run all. Old artifacts stay intact.

## Phased rollout — smoke first, then full

The notebook has a top-of-cell variable `SMOKE_MAX_EPOCH` that controls
training duration:

```python
SMOKE_MAX_EPOCH = 11    # First run: ~15 min on A100, exercises full pipeline
# SMOKE_MAX_EPOCH = None  # Subsequent runs: paper recipe (101 epochs from hparams)
```

### First run (smoke, ~15 minutes on A100)

1. Open notebook in Colab (steps 1–3 above)
2. Confirm `SMOKE_MAX_EPOCH = 11` (it's the default — you don't need to change it)
3. `Runtime` → `Run all`
4. After ~15 min, verify in cell output:
   - `==> Final artifacts:` line appears
   - `best_jazz_model_{8|16}bars.pth.tar` is listed at the end (depending on `HPARAMS`)
5. Open Drive web UI, navigate to `cmt-training/result_{8|16}bars/`, confirm the file exists

If smoke fails, copy the traceback from the cell output and report it.

### Full run (~1–2 hours on A100, longer on V100/T4)

1. After successful smoke, **clean `result/`** in Drive (so phases re-train
   from scratch instead of resuming from epoch 10):
   ```
   In a separate cell or in Drive web UI:
   !rm -rf /content/drive/MyDrive/cmt-training/result
   ```
2. Edit the notebook cell — change `SMOKE_MAX_EPOCH = 11` to `SMOKE_MAX_EPOCH = None`
3. `Runtime` → `Run all`
4. Watch initial progress for 2 epochs — `[ETA]` lines appear in cell
   output once enough timestamps are recorded
5. Close the browser tab — Pro+ background execution keeps the runtime running
6. Reopen the tab periodically (or after a few hours) to see progress
7. When `==> Final artifacts:` appears, training is done

## Reading progress

The cell output mixes two streams:

- **Trainer messages** (one per epoch): `==========valid 23 epoch==========`,
  `nll/eval: ...`, `accuracy_rhythm/eval: ...`
- **`[ETA]` lines** (every 60 seconds, from background daemon thread):
  `[ETA] Phase 1: epoch 23/101, 42s/epoch, ETA 55 min`

The ETA is computed from **measured timestamps** in `log.txt` — there are no
hardcoded duration estimates. Whatever GPU Colab allocated to you,
the ETA reflects its actual speed.

## Resume after disconnect

If the runtime disconnects mid-training:

1. Reopen the notebook in Colab
2. `Runtime` → `Run all`
3. The orchestrator (`training.train`) detects checkpoints from previous
   run, resumes Phase 1 or Phase 2 from the last saved epoch
4. ETA monitor continues from where it left off

No manual cleanup or reconfiguration needed.

## Verifying GPU allocation

In a separate cell (after Mount Drive is done), run:

```python
!nvidia-smi
```

Look for the GPU model in the output (e.g. `Tesla A100-SXM4-40GB`).
If you got T4 or V100 and want A100, you can disconnect and reconnect
the runtime — Colab's allocator reshuffles. No guarantee, A100 depends
on demand.

## Cost (compute units)

- A100: ~10–15 units/hour
- V100: ~5–10 units/hour
- T4: ~2–4 units/hour
- Pro+ provides 500 units/month → easily fits multiple full runs

You can monitor units left in the Colab UI: `Tools` → `Manage sessions`.

## Failure modes

| Symptom | Action |
|---|---|
| `pytest` fails in cell | Read test output, fix locally, `git push origin master`, re-run |
| `run.py` traceback (e.g. CUDA OOM) | Lower `data_io.loader.batch_size` in your active `hparams_jazz_{8|16}bars.yaml`, push, re-run |
| Drive mount times out | Re-run cell; if persistent, sign in to Google in another tab first |
| Compute units exhausted | Wait for monthly reset or buy units |
| 24-hour session limit hit | Re-run notebook next day; orchestrator resumes from last checkpoint |
| Notebook URL gives 404 | `master` not pushed yet — run `git push origin master` from local repo |

## Getting the trained model back

After training completes, `best_jazz_model_{N}bars.pth.tar` (~147 MB) is in:
`/content/drive/MyDrive/cmt-training/result_{N}bars/best_jazz_model_{N}bars.pth.tar`
(where `N` is `8` or `16` depending on selected `HPARAMS`).

To use it in the diploma pipeline:

1. Sync Drive to your local machine (Google Drive desktop app), OR
   download the file via Drive web UI manually
2. Place into `models/CMT-pytorch/result/paper/{N}bars/` (or wherever your
   pipeline config expects)
3. Update `pipeline/pipeline/config.py::CMT_CHECKPOINT_PATH` to point
   at the new file

This step is outside Plan 3 scope — manual integration.
