# WJazzD → CMT MIDI

Reproducible pipeline that converts raw Weimar Jazz Database (WJazzD r2.2) into
2-track MIDI files compatible with the authorial CMT `preprocess.py`.

## Quickstart

```bash
# 1. Setup venv (Python 3.12)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-data-prep.txt

# 2. Download wjazzd.db (~42 MB) from jazzomat.hfm-weimar.de.
#    The DB is also committed to git (jazz/wjazzd/data/wjazzd.db) — this
#    step is only needed for fresh setups or manual re-download.
.venv/bin/python -m jazz.wjazzd.dataset_converter.download

# 3. Convert all eligible solos to 2-track MIDI
.venv/bin/python -m jazz.wjazzd.dataset_converter.wjazzd_to_cmt_midi
# → 430 MIDI files in jazz/wjazzd/data/midi/<melid>_<performer>_<title>_Solo/

# 4. Run authorial preprocess.py to generate .pkl training instances
.venv/bin/pip install numpy scipy tqdm
.venv/bin/python preprocess.py \
    --root_dir jazz/wjazzd/data --midi_dir midi \
    --num_bars 8 --stride_bars 4 --frame_per_bar 16 --pitch_range 48
# → ~4178 pkl files split into train/eval/test in jazz/wjazzd/data/pkl_files/
```

Add `--shift` to the last command to also produce 12-keys augmented dataset
(needed for Phase 1 of CMT training — Rhythm Decoder).

## What gets filtered

Of 456 solos in WJazzD, `jazz/wjazzd/data/midi/` contains 430 (94%):

- **21 solos** dropped due to non-4/4 signature (3/4, 5/4, 6/8, 9/4, or unspecified)
- **5 solos** dropped due to internal signature change within solo (4/4 + 5/4 etc)

CMT preprocess.py is hard-coded to 4/4 (`beat_per_bar=4`) so non-4/4 solos cannot
be used. See `docs/cmt-jazz-training-brainstorm.md` Решение 16 for details.

## Architecture

```
jazz/wjazzd/
├── dataset_converter/             # converter code
│   ├── chord_parser.py            # jazzomat chord string → MIDI pitch classes
│   ├── filter.py                  # solo eligibility (signature, length)
│   ├── wjazzd_to_cmt_midi.py      # main: load_solo → build MIDI → save
│   ├── download.py                # idempotent wjazzd.db fetch
│   └── tests/                     # unit + regression tests
└── data/                          # data artefacts
    ├── wjazzd.db                  # 42 MB, committed (raw source)
    ├── midi/                      # 430 MIDI files (2-track: melody + chord), committed
    └── pkl_files/                 # CMT preprocess output (gitignored)
```

### Chord parser

Pure-Python parser for jazzomat chord notation. ~99.81% parse rate on full
WJazzD corpus (30090 of 30147 chord beat occurrences successfully parsed).

Supports: triads, sevenths, alterations (b9/#9/#11/b13), `alt` expansion, sus
chords, slash chords (bass stripped — bass comes from `beats.bass_pitch`),
jazzomat shorthands `j` (=maj), `-` (=minor), `°` (=dim), `ø` (=m7b5).

**`music21.harmony.ChordSymbol` is NOT used** — it's incompatible with jazzomat
syntax (treats `b` as flat-7 alteration not flat-root, doesn't know `j`/`-`).
See `docs/cmt-jazz-training-brainstorm.md` Решение 7a for empirical analysis.

### Output MIDI layout

Per Решение 12 (nested directory required by authorial preprocess.py — uses
parent directory name as `song_title` for train/eval/test split):

```
jazz/wjazzd/data/midi/{melid:03d}_{performer}_{title}_Solo/
                                       └── {melid:03d}_{performer}_{title}_Solo.mid
```

### Pitch-class set (chord-track)

Per Решение 7b ("variant C"): chord-tones + alterations only. Natural extensions
(9, 11, 13 without `#`/`b`) do NOT add new pitch classes — `C7` ≡ `C9` ≡ `C13`
in the output. This is a deliberate trade-off losing color information for
parser simplicity.

## Patches to authorial `preprocess.py`

We made several changes to make the script usable for our jazz pipeline.
None alter the model output for well-formed inputs — they extend the CLI,
fix a crash on jazz-specific edge cases, and improve Colab observability.

1. **`--stride_bars` CLI argument.** Previously the sliding-window stride
   was hardcoded to `num_bars // 2`. Made it a CLI parameter so 8/16/32-bar
   configs can each use the appropriate stride (we keep `stride=4` for all).

2. **Output folder name comes from `training/pkl_paths.py`.** Replaced
   the hardcoded folder string with a call to `pkl_dir_name(...)` so the
   on-disk structure is the single source of truth shared with the
   training orchestrator.

3. **Empty-onset guard** (at line ~118, after the `rhythm_idx` check):
   ```python
   if onset_inst.nonzero()[1].size == 0:
       continue
   ```
   Fixes `ValueError: max() iterable argument is empty` on jazz solos
   that produce sliding windows containing only sustained notes from
   prior bars (no fresh onsets). Authors didn't hit this on their LMD
   dataset.

4. **`print(...)` instead of `tqdm`** for progress every 25 files.
   `tqdm` writes carriage-return updates to stderr; Jupyter/Colab
   buffers stderr separately and frequently swallows the CR, so the
   bar appears frozen. Plain `print(..., flush=True)` survives Colab.

5. **Startup print before heavy imports** so a stuck process is
   immediately visible (otherwise an import hang produces no output
   for 10+ seconds and looks like total silence).

6. **Extracted `extract_instances_from_midi()` as a standalone function.**
   The inner per-window logic of `make_instance_pkl_files` was inlined
   originally. Pulled it out so future inference code can call it
   directly without round-tripping to disk.

## Tests

```bash
.venv/bin/python -m pytest jazz/wjazzd/dataset_converter/tests/ -v
```

102 tests across:

- `test_chord_parser.py` — unit tests for parser (60 cases)
- `test_chord_corpus.py` — regression on all unique chord strings in wjazzd.db
- `test_download.py` — idempotency tests
- `test_filter.py` — eligibility logic
- `test_convert.py` — load_solo, _beat_to_sec, chord_changes, build_*

## License

WJazzD is released under [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/).
The committed copy in `data/wjazzd.db` is included for reproducibility under
ODbL terms; `dataset_converter/download.py` can also fetch it fresh from
[jazzomat.hfm-weimar.de](https://jazzomat.hfm-weimar.de).

Code in this directory is part of the CMT-pytorch fork and inherits its license.
