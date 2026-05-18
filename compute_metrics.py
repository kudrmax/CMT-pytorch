"""Compute paper-style metrics on test_inference output (162 instances).

Each pkl in --inference-dir contains:
- pitch, rhythm: model output (T=129)
- groundtruth_pitch, groundtruth_rhythm: real human melody (same chord)
- chord: 12xT pitch class binary vectors (same for both)

Metrics (per Choi et al. 2021, Tables 2 + 4):
1. Chord-tone ratio (overall + 1st-beat)
2. Pitch Class Histogram (PCH) — KL divergence + Overlapping Area
3. Note Count mean
4. Inter-Onset Interval (IOI) — distribution distance
5. Pitch Range mean
6. Bar Rhythm distribution — Jensen-Shannon divergence
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

# Frame encoding (matches authorial preprocess.py)
PITCH_HOLD = 48      # tied note continues
PITCH_REST = 49      # silence
RHYTHM_REST = 0
RHYTHM_HOLD = 1
RHYTHM_ONSET = 2


def find_chord_root(chord_pcs):
    """Heuristic: root = pc with both a 3rd (m or M) and a 5th (P, b, or aug) active."""
    for root in chord_pcs:
        has_third = (root + 3) % 12 in chord_pcs or (root + 4) % 12 in chord_pcs
        has_fifth = any((root + s) % 12 in chord_pcs for s in (6, 7, 8))
        if has_third and has_fifth:
            return root
    return min(chord_pcs)  # fallback


def chord_to_scale(chord_pcs):
    """Map chord pitch-class set → diatonic scale (set of 7 pitch classes).

    Jazz convention chord-scale relationships:
      - m7b5 → Locrian
      - m7    → Dorian
      - minor → natural minor
      - dom7  → Mixolydian
      - maj7  → Ionian
      - default → Ionian
    """
    if not chord_pcs:
        return set(range(12))  # any pitch valid if no chord
    root = find_chord_root(chord_pcs)
    has_m3 = (root + 3) % 12 in chord_pcs
    has_M3 = (root + 4) % 12 in chord_pcs
    has_b5 = (root + 6) % 12 in chord_pcs
    has_m7 = (root + 10) % 12 in chord_pcs
    has_M7 = (root + 11) % 12 in chord_pcs

    if has_m3 and has_b5:
        intervals = [0, 1, 3, 5, 6, 8, 10]   # Locrian
    elif has_m3 and has_m7:
        intervals = [0, 2, 3, 5, 7, 9, 10]   # Dorian
    elif has_m3:
        intervals = [0, 2, 3, 5, 7, 8, 10]   # natural minor
    elif has_M3 and has_m7:
        intervals = [0, 2, 4, 5, 7, 9, 10]   # Mixolydian
    elif has_M3 and has_M7:
        intervals = [0, 2, 4, 5, 7, 9, 11]   # Ionian
    else:
        intervals = [0, 2, 4, 5, 7, 9, 11]   # default Ionian
    return {(root + i) % 12 for i in intervals}


def scale_match_ratio(pitch_arr, rhythm_arr, chord_arr):
    """Fraction of melody onsets whose pitch class belongs to the chord-scale."""
    onsets = np.where(rhythm_arr == RHYTHM_ONSET)[0]
    if len(onsets) == 0:
        return None, 0
    in_scale, total = 0, 0
    for t in onsets:
        p = pitch_arr[t]
        if p >= PITCH_HOLD:
            continue
        pc = int(p) % 12
        chord_pcs = set(np.where(chord_arr[t] > 0)[0].tolist())
        if not chord_pcs:
            continue
        scale = chord_to_scale(chord_pcs)
        total += 1
        if pc in scale:
            in_scale += 1
    return (in_scale / total if total else None), total


def pitch_entropy(pitch_arr, rhythm_arr):
    """Shannon entropy (in bits) of pitch-class distribution across onsets.

    Range: 0 (one pitch only) to log2(12) ≈ 3.585 (uniform across 12 classes).
    Higher = more chromatic / diverse pitch usage.
    """
    onsets = np.where(rhythm_arr == RHYTHM_ONSET)[0]
    pcs = [int(pitch_arr[t]) % 12 for t in onsets if pitch_arr[t] < PITCH_HOLD]
    if not pcs:
        return 0.0
    counts = Counter(pcs)
    probs = np.array(list(counts.values()), dtype=float) / len(pcs)
    return float(-np.sum(probs * np.log2(probs + 1e-9)))


def extract_notes(pitch_arr, rhythm_arr):
    """Return list of (onset_frame, pitch, duration_frames) for actual notes."""
    notes = []
    onset = None
    pitch = None
    for t, (p, r) in enumerate(zip(pitch_arr, rhythm_arr)):
        if r == RHYTHM_ONSET and p < PITCH_HOLD:
            if onset is not None:
                notes.append((onset, pitch, t - onset))
            onset, pitch = t, int(p)
        elif r == RHYTHM_REST and onset is not None:
            notes.append((onset, pitch, t - onset))
            onset, pitch = None, None
    if onset is not None:
        notes.append((onset, pitch, len(pitch_arr) - onset))
    return notes


def pitch_interval_entropy(pitch_arr, rhythm_arr):
    """Entropy of melodic interval distribution (semitones, signed)."""
    notes = extract_notes(pitch_arr, rhythm_arr)
    if len(notes) < 2:
        return 0.0
    intervals = [notes[i+1][1] - notes[i][1] for i in range(len(notes)-1)]
    counts = Counter(intervals)
    probs = np.array(list(counts.values()), dtype=float) / len(intervals)
    return float(-np.sum(probs * np.log2(probs + 1e-9)))


def note_duration_entropy(pitch_arr, rhythm_arr):
    notes = extract_notes(pitch_arr, rhythm_arr)
    if not notes:
        return 0.0
    durs = [n[2] for n in notes]
    counts = Counter(durs)
    probs = np.array(list(counts.values()), dtype=float) / len(durs)
    return float(-np.sum(probs * np.log2(probs + 1e-9)))


def avg_note_duration(pitch_arr, rhythm_arr):
    notes = extract_notes(pitch_arr, rhythm_arr)
    return float(np.mean([n[2] for n in notes])) if notes else 0.0


def rest_ratio(rhythm_arr):
    """Fraction of frames that are rests."""
    return float(np.sum(rhythm_arr == RHYTHM_REST) / len(rhythm_arr))


def repetition_ratio(pitch_arr, rhythm_arr):
    """Fraction of consecutive notes with the same pitch (immediate repeats)."""
    notes = extract_notes(pitch_arr, rhythm_arr)
    if len(notes) < 2:
        return 0.0
    same = sum(1 for i in range(len(notes)-1) if notes[i][1] == notes[i+1][1])
    return same / (len(notes) - 1)


def direction_changes(pitch_arr, rhythm_arr):
    """Number of melodic contour direction changes per note."""
    notes = extract_notes(pitch_arr, rhythm_arr)
    if len(notes) < 3:
        return 0.0
    signs = [np.sign(notes[i+1][1] - notes[i][1]) for i in range(len(notes)-1)]
    changes = sum(1 for i in range(len(signs)-1)
                  if signs[i] != 0 and signs[i+1] != 0 and signs[i] != signs[i+1])
    return changes / (len(notes) - 2)


def step_vs_leap_ratio(pitch_arr, rhythm_arr):
    """Fraction of intervals ≤2 semitones (stepwise motion). Smooth jazz lines tend high."""
    notes = extract_notes(pitch_arr, rhythm_arr)
    if len(notes) < 2:
        return None
    intervals = [abs(notes[i+1][1] - notes[i][1]) for i in range(len(notes)-1)]
    return sum(1 for x in intervals if x <= 2) / len(intervals)


def chromatic_ratio(pitch_arr, rhythm_arr, chord_arr):
    """Fraction of onsets OUTSIDE chord-scale (= 1 − scale_match, but more direct interpretation)."""
    sm, total = scale_match_ratio(pitch_arr, rhythm_arr, chord_arr)
    if sm is None:
        return None
    return 1.0 - sm


def chord_tone_ratio(pitch_arr, rhythm_arr, chord_arr, only_first_beat=False, frame_per_bar=16):
    """Fraction of melody onsets whose pitch class is in current chord.

    only_first_beat=True restricts to frames at start of each bar (paper Table 2).
    """
    onsets = np.where(rhythm_arr == RHYTHM_ONSET)[0]
    if only_first_beat:
        onsets = onsets[onsets % frame_per_bar == 0]
    if len(onsets) == 0:
        return None, 0
    # chord_arr shape: (T, 12) — frame-major
    in_chord, total = 0, 0
    for t in onsets:
        p = pitch_arr[t]
        if p >= PITCH_HOLD:  # not a real pitch token
            continue
        pc = int(p) % 12
        chord_pcs = np.where(chord_arr[t] > 0)[0]
        if len(chord_pcs) == 0:
            continue
        total += 1
        if pc in chord_pcs:
            in_chord += 1
    return (in_chord / total if total else None), total


def pitch_class_histogram(pitch_arr, rhythm_arr):
    """Frequency vector of 12 pitch classes across onsets (sums to 1)."""
    onsets = np.where(rhythm_arr == RHYTHM_ONSET)[0]
    pcs = [int(pitch_arr[t]) % 12 for t in onsets if pitch_arr[t] < PITCH_HOLD]
    if not pcs:
        return np.ones(12) / 12  # uniform if empty
    h = np.bincount(pcs, minlength=12).astype(float)
    return h / h.sum()


def kl_divergence(p, q, eps=1e-9):
    """KL(p || q). Inputs are distributions (sum to 1)."""
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def overlapping_area(p, q):
    """Sum of element-wise minimum (overlap area between two distributions)."""
    return float(np.sum(np.minimum(p, q)))


def jensen_shannon(p, q, eps=1e-9):
    """Symmetric JS divergence."""
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def note_count(rhythm_arr):
    return int(np.sum(rhythm_arr == RHYTHM_ONSET))


def inter_onset_intervals(rhythm_arr):
    """Frame-distances between consecutive onsets."""
    onsets = np.where(rhythm_arr == RHYTHM_ONSET)[0]
    return np.diff(onsets) if len(onsets) > 1 else np.array([])


def pitch_range(pitch_arr, rhythm_arr):
    onsets = np.where(rhythm_arr == RHYTHM_ONSET)[0]
    pitches = [int(pitch_arr[t]) for t in onsets if pitch_arr[t] < PITCH_HOLD]
    if not pitches:
        return 0
    return max(pitches) - min(pitches)


def bar_rhythm_pattern(rhythm_arr, frame_per_bar=16):
    """For each bar: tuple of rhythm tokens. Used for distribution comparison."""
    n_bars = len(rhythm_arr) // frame_per_bar
    return [tuple(rhythm_arr[i*frame_per_bar:(i+1)*frame_per_bar]) for i in range(n_bars)]


def aggregate_pch(pchs):
    """Average pitch-class histogram across instances."""
    if not pchs:
        return np.ones(12) / 12
    return np.mean(pchs, axis=0)


def aggregate_ioi_dist(iois_list, max_interval=32):
    """Histogram of IOIs across instances, bins 1..max_interval."""
    counter = Counter()
    for iois in iois_list:
        for v in iois:
            if 1 <= v <= max_interval:
                counter[v] += 1
    total = sum(counter.values()) or 1
    h = np.array([counter.get(i, 0) / total for i in range(1, max_interval + 1)])
    return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--frame-per-bar", type=int, default=16)
    args = parser.parse_args()

    pkl_files = sorted(args.inference_dir.glob("*.pkl"))
    print(f"Loaded {len(pkl_files)} test instances\n")

    # Per-instance metric collectors
    rows = {"model": {}, "gt": {}}
    for k in rows:
        rows[k] = {
            "ct_ratio": [], "ct_ratio_1st": [],
            "scale_match": [], "pitch_entropy": [],
            "interval_entropy": [], "duration_entropy": [],
            "avg_dur": [], "rest_ratio": [],
            "repetition": [], "dir_changes": [], "step_ratio": [],
            "pch": [], "note_count": [], "ioi": [],
            "pitch_range": [], "bar_patterns": [],
        }

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            inst = pickle.load(f)
        chord = inst["chord"].toarray() if hasattr(inst["chord"], "toarray") else inst["chord"]

        for kind, pkey, rkey in [("model", "pitch", "rhythm"),
                                   ("gt", "groundtruth_pitch", "groundtruth_rhythm")]:
            p = np.asarray(inst[pkey])
            r = np.asarray(inst[rkey])

            ct, _ = chord_tone_ratio(p, r, chord, only_first_beat=False, frame_per_bar=args.frame_per_bar)
            ct1, _ = chord_tone_ratio(p, r, chord, only_first_beat=True, frame_per_bar=args.frame_per_bar)
            sm, _ = scale_match_ratio(p, r, chord)
            if ct is not None: rows[kind]["ct_ratio"].append(ct)
            if ct1 is not None: rows[kind]["ct_ratio_1st"].append(ct1)
            if sm is not None: rows[kind]["scale_match"].append(sm)
            rows[kind]["pitch_entropy"].append(pitch_entropy(p, r))
            rows[kind]["interval_entropy"].append(pitch_interval_entropy(p, r))
            rows[kind]["duration_entropy"].append(note_duration_entropy(p, r))
            rows[kind]["avg_dur"].append(avg_note_duration(p, r))
            rows[kind]["rest_ratio"].append(rest_ratio(r))
            rows[kind]["repetition"].append(repetition_ratio(p, r))
            rows[kind]["dir_changes"].append(direction_changes(p, r))
            sl = step_vs_leap_ratio(p, r)
            if sl is not None: rows[kind]["step_ratio"].append(sl)

            rows[kind]["pch"].append(pitch_class_histogram(p, r))
            rows[kind]["note_count"].append(note_count(r))
            rows[kind]["ioi"].append(inter_onset_intervals(r))
            rows[kind]["pitch_range"].append(pitch_range(p, r))
            rows[kind]["bar_patterns"].extend(bar_rhythm_pattern(r, args.frame_per_bar))

    # ===== Aggregate =====
    print("=" * 60)
    print("Chord-tone ratio (paper Table 2)")
    print("=" * 60)
    print(f"{'':10} {'overall':>10} {'1st beat':>10}  vs paper")
    print(f"{'model':10} {mean(rows['model']['ct_ratio']):>10.3f} {mean(rows['model']['ct_ratio_1st']):>10.3f}  CMT paper: 0.725 / 0.806")
    print(f"{'gt':10} {mean(rows['gt']['ct_ratio']):>10.3f} {mean(rows['gt']['ct_ratio_1st']):>10.3f}  EWLD GT:   0.714 / 0.796")

    print(f"\n{'=' * 60}")
    print("Pitch Class Histogram (paper Table 4 'PCH')")
    print("=" * 60)
    pch_m = aggregate_pch(rows["model"]["pch"])
    pch_g = aggregate_pch(rows["gt"]["pch"])
    print(f"KL(model||gt) = {kl_divergence(pch_m, pch_g):.4f}  (paper CMT KL = 1.87e-3, EC2-VAE = 5.42e-1)")
    print(f"Overlapping area = {overlapping_area(pch_m, pch_g):.4f}  (paper CMT OA = 0.983)")

    print(f"\n{'=' * 60}")
    print("Note Count per 8-bar window (paper Table 4 'NC')")
    print("=" * 60)
    nc_m = mean(rows["model"]["note_count"])
    nc_g = mean(rows["gt"]["note_count"])
    print(f"model mean = {nc_m:.2f}    gt mean = {nc_g:.2f}    diff = {abs(nc_m - nc_g):.2f}")
    print(f"(paper EWLD GT mean = 28.5; CMT mean = 27.4)")

    print(f"\n{'=' * 60}")
    print("Inter-Onset Interval (paper Table 4 'IOI', frames)")
    print("=" * 60)
    ioi_m = aggregate_ioi_dist(rows["model"]["ioi"])
    ioi_g = aggregate_ioi_dist(rows["gt"]["ioi"])
    print(f"KL(model||gt) = {kl_divergence(ioi_m, ioi_g):.4f}  (paper CMT KL = 3.95e-2)")
    print(f"Overlapping area = {overlapping_area(ioi_m, ioi_g):.4f}  (paper CMT OA = 0.979)")
    print(f"mean IOI: model = {mean([float(np.mean(x)) if len(x) else 0 for x in rows['model']['ioi']]):.2f} frames; gt = {mean([float(np.mean(x)) if len(x) else 0 for x in rows['gt']['ioi']]):.2f}")

    print(f"\n{'=' * 60}")
    print("Pitch Range (paper Table 4 'PR', semitones)")
    print("=" * 60)
    print(f"model mean = {mean(rows['model']['pitch_range']):.2f}    gt mean = {mean(rows['gt']['pitch_range']):.2f}")
    print(f"(paper EWLD GT mean = 12.26; CMT mean = 11.51)")

    # KL/OA on pitch-range distribution — compatible with MINGUS Table 3 'pitch range'
    pr_max = 60
    def _pr_dist(values, n=pr_max):
        h = np.bincount(np.clip(values, 0, n-1), minlength=n).astype(float)
        return h / (h.sum() or 1)
    pr_m = _pr_dist(rows['model']['pitch_range'])
    pr_g = _pr_dist(rows['gt']['pitch_range'])
    print(f"KL(model||gt) = {kl_divergence(pr_m, pr_g):.4f}    OA = {overlapping_area(pr_m, pr_g):.4f}")
    print(f"(MINGUS paper Table 3: MINGUS=0.037/0.844, BebopNet=0.093/0.571)")

    print(f"\n{'=' * 60}")
    print("Bar Rhythm Patterns (paper Section V-A-4)")
    print("=" * 60)
    pat_m = Counter(rows["model"]["bar_patterns"])
    pat_g = Counter(rows["gt"]["bar_patterns"])
    all_patterns = list(set(pat_m.keys()) | set(pat_g.keys()))
    n_m, n_g = sum(pat_m.values()), sum(pat_g.values())
    p_m = np.array([pat_m.get(p, 0) / n_m for p in all_patterns])
    p_g = np.array([pat_g.get(p, 0) / n_g for p in all_patterns])
    js = jensen_shannon(p_m, p_g)
    print(f"Unique patterns:  model = {len(pat_m)},  gt = {len(pat_g)},  union = {len(all_patterns)}")
    print(f"Jensen-Shannon divergence = {js:.4f}  (paper CMT JS = 8.14e-2; EC2-VAE = 2.17e-1)")

    print(f"\n{'=' * 60}")
    print("Jazz-specific metrics (project proposal)")
    print("=" * 60)

    def _row(label, key, fmt=".3f"):
        m = mean(rows["model"][key]) if rows["model"][key] else float("nan")
        g = mean(rows["gt"][key]) if rows["gt"][key] else float("nan")
        delta = m - g
        print(f"{label:30}  model={m:{fmt}}   gt={g:{fmt}}   Δ={delta:+{fmt}}")

    _row("Scale match ratio",        "scale_match")
    _row("Pitch entropy (bits)",     "pitch_entropy")
    _row("Interval entropy (bits)",  "interval_entropy")
    _row("Note duration entropy",    "duration_entropy")
    _row("Avg note duration (frames)","avg_dur",       fmt=".2f")
    _row("Rest ratio",               "rest_ratio")
    _row("Immediate-repetition rate","repetition")
    _row("Direction changes / note", "dir_changes")
    _row("Step (≤2 semi) ratio",     "step_ratio")

    print()
    print("Hint: closer to gt is better. Pitch/interval/duration entropy")
    print("indicate variety; rest_ratio + avg_dur shape the rhythmic feel;")
    print("repetition + dir_changes + step_ratio shape the melodic contour.")


if __name__ == "__main__":
    main()
