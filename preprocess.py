import sys
print("[preprocess] starting up (importing deps)...", flush=True)
sys.stdout.flush()
import glob
import json
import os
import random
import argparse
import pickle
import numpy as np
import pretty_midi as pm
from tqdm import tqdm
from scipy.sparse import csc_matrix
print("[preprocess] deps loaded", flush=True)
sys.stdout.flush()


def pad_pianorolls(pianoroll, timelen):
    if pianoroll.shape[1] < timelen:
        pianoroll = np.pad(pianoroll, ((0, 0), (0, timelen - pianoroll.shape[1])),
                           mode="constant", constant_values=0)
    return pianoroll


def extract_instances_from_midi(midi_path, num_bars, frame_per_bar, stride_bars,
                                 pitch_range=48, k_shift=0,
                                 beat_per_bar=4, bpm=120):
    """Process one 2-track MIDI into per-window instance dicts.

    Returns a list of dicts with keys
        'pitch'       : np.ndarray (instance_len + 1,) int — onset/hold/rest tokens
        'rhythm'      : np.ndarray (instance_len + 1,) int — 0=rest, 1=hold, 2=onset
        'chord'       : scipy.sparse.csc_matrix (instance_len + 1, 12) — pitch classes
        'window_idx'  : int — i // stride (= the suffix make_instance_pkl_files uses for filenames)
        'base_note'   : int — MIDI offset such that pitch token X means MIDI (X + base_note)

    Empty list if every sliding-window position fails one of the filters
    (chord-count, rhythm-activity, empty-onset guard, pitch-range overflow,
    contiguous-rest break, or pitch-variety).

    The function is the inner per-window logic of make_instance_pkl_files
    extracted as a standalone callable; inference wrappers can call it
    directly to skip the disk pkl detour.
    """
    instance_len = frame_per_bar * num_bars
    stride = stride_bars * frame_per_bar
    # Default : frame_per_second=8, unit_time=0.125
    frame_per_second = (frame_per_bar / beat_per_bar) * (bpm / 60)
    unit_time = 1 / frame_per_second

    midi = pm.PrettyMIDI(midi_path)
    if len(midi.instruments) < 2:
        return []
    on_midi = pm.PrettyMIDI(midi_path)
    off_midi = pm.PrettyMIDI(midi_path)
    note_instrument = midi.instruments[0]
    onset_instrument = on_midi.instruments[0]
    offset_instrument = off_midi.instruments[0]
    for note, onset_note, offset_note in zip(note_instrument.notes, onset_instrument.notes, offset_instrument.notes):
        if k_shift != 0:
            note.pitch += k_shift
            onset_note.pitch += k_shift
            offset_note.pitch += k_shift
        note_length = offset_note.end - offset_note.start
        onset_note.end = onset_note.start + min(note_length, unit_time)
        offset_note.end += unit_time
        offset_note.start = offset_note.end - min(note_length, unit_time)
    pianoroll = note_instrument.get_piano_roll(fs=frame_per_second)
    onset_roll = onset_instrument.get_piano_roll(fs=frame_per_second)
    offset_roll = offset_instrument.get_piano_roll(fs=frame_per_second)

    chord_instrument = midi.instruments[1]
    timelen = min(pianoroll.shape[1], offset_roll.shape[1])
    for chord_note in chord_instrument.notes:
        if k_shift != 0:
            chord_note.pitch += k_shift
        chord_note.end = chord_note.start + unit_time
    chord_onset = chord_instrument.get_piano_roll(fs=frame_per_second)

    pianoroll = pad_pianorolls(pianoroll, timelen)
    onset_roll = pad_pianorolls(onset_roll, timelen)
    offset_roll = pad_pianorolls(offset_roll, timelen)
    chord_onset = pad_pianorolls(chord_onset, timelen)

    pianoroll[pianoroll > 0] = 1
    onset_roll[onset_roll > 0] = 1
    offset_roll[offset_roll > 0] = 1
    chord_onset[chord_onset > 0] = 1

    instances = []
    for i in range(0, timelen - (instance_len + 1), stride):
        pitch_list = []
        chord_list = []

        pianoroll_inst = pianoroll[:, i:(i+instance_len+1)]
        onset_inst = onset_roll[:, i:(i+instance_len+1)]
        chord_inst = chord_onset[:, i:(i + instance_len + 1)]

        if len(chord_inst.nonzero()[1]) < 4:
            continue

        rhythm_idx = np.minimum(np.sum(pianoroll_inst.T, axis=1), 1) + np.minimum(np.sum(onset_inst.T, axis=1), 1)
        rhythm_idx = rhythm_idx.astype(int)
        # If more than 75% is not-playing, do not make instance
        if rhythm_idx.nonzero()[0].size < (instance_len // 4):
            continue

        # Guard: skip windows with no onsets (e.g. only sustained notes from prior bars)
        if onset_inst.nonzero()[1].size == 0:
            continue
        if pitch_range == 128:
            base_note = 0
        else:
            highest_note = max(onset_inst.T.nonzero()[1])
            lowest_note = min(onset_inst.T.nonzero()[1])
            base_note = 12 * (lowest_note // 12)
            if highest_note - base_note >= pitch_range:
                continue

        prev_chord = np.zeros(12)
        cont_rest = 0
        prev_onset = 0
        for t in range(instance_len+1):
            if t in onset_inst.T.nonzero()[0]:
                pitch_list.append(onset_inst[:, t].T.nonzero()[0][0] - base_note)
                if (t != onset_inst.T.nonzero()[0][0]) and abs(onset_inst[:, t].T.nonzero()[0][0] - base_note - prev_onset) > 12:
                    cont_rest = 30
                    break
                else:
                    prev_onset = onset_inst[:, t].T.nonzero()[0][0] - base_note
                    cont_rest = 0
            elif rhythm_idx[t] == 1:
                pitch_list.append(pitch_range)
            elif rhythm_idx[t] == 0:
                pitch_list.append(pitch_range + 1)
                cont_rest += 1
                if cont_rest >= 30:
                    break
            else:
                print(midi_path, i, t, rhythm_idx[t], onset_inst.T.nonzero())

            if len(chord_inst[:, t].nonzero()[0]) != 0:
                prev_chord = np.zeros(12)
                for note in sorted(chord_inst[:, t].nonzero()[0][1:] % 12):
                    prev_chord[note] = 1
            chord_list.append(prev_chord)

        if (cont_rest >= 30) or (len(set(pitch_list)) <= 5):
            continue

        pitch_list = np.array(pitch_list)
        chord_result = csc_matrix(np.array(chord_list))
        instances.append({
            'pitch': pitch_list,
            'rhythm': rhythm_idx,
            'chord': chord_result,
            'window_idx': i // stride,
            'base_note': int(base_note),
        })

    return instances


def make_instance_pkl_files(root_dir, midi_dir, num_bars, frame_per_bar, stride_bars,
                            pitch_range=48, shift=False,
                            beat_per_bar=4, bpm=120, data_ratio=(0.8, 0.1, 0.1),
                            split_json=None):
    from training.pkl_paths import pkl_dir_name
    instance_folder = pkl_dir_name(
        num_bars=num_bars, stride_bars=stride_bars,
        frame_per_bar=frame_per_bar, pitch_range=pitch_range, shift=shift,
    )

    dir_name = os.path.join(root_dir, 'pkl_files', instance_folder)
    os.makedirs(dir_name, exist_ok=True)

    song_list = sorted(glob.glob(os.path.join(root_dir, midi_dir, '*')))
    midi_files = sorted(glob.glob(os.path.join(root_dir, midi_dir, '*/*.mid')))

    if split_json:
        # Cross-model SSoT: read bucket assignments from
        # diploma2/pipelines/training-pipeline/wjazzd_split.json. Train/eval
        # are bit-exact identical to the legacy random.seed(0) sample below
        # (split.json was generated from the same algorithm). Test is 40 files
        # (3 MINGUS-incompatible files removed from the base 43); those 3
        # files end up in no bucket and are skipped during pkl generation.
        with open(split_json) as _f:
            _split = json.load(_f)
        train_set = set(_split["train"])
        eval_set = set(_split["eval"])
        test_set = set(_split["test"])
        excluded_from_all = (
            set(s.split('/')[-1] for s in song_list)
            - train_set - eval_set - test_set
        )
        if excluded_from_all:
            print(f"[preprocess] skipping {len(excluded_from_all)} files not in any "
                  f"split.json bucket: {sorted(excluded_from_all)}", flush=True)
    else:
        # Legacy fallback (kept for back-compat): authorial random.seed(0) sample.
        # Bit-exact identical to cmt_base_split() in diploma2/generate_split.py,
        # so existing on-disk pkl produced this way are on the canonical split.
        train_set = None  # not used in this path
        num_eval = int(len(song_list) * data_ratio[1])
        num_test = int(len(song_list) * data_ratio[2])
        random.seed(0)
        eval_test_cand = set([song.split('/')[-1] for song in song_list])
        eval_set = set(random.sample(sorted(eval_test_cand), num_eval))
        test_set = set(random.sample(sorted(eval_test_cand - eval_set), num_test))

    print(f"[preprocess] starting: {len(midi_files)} files, shift={shift}, num_bars={num_bars}", flush=True)
    _t0 = __import__('time').time()
    for _i, midi_file in enumerate(midi_files):
        if _i % 25 == 0 or _i == len(midi_files) - 1:
            _elapsed = __import__('time').time() - _t0
            print(f"[preprocess] {_i}/{len(midi_files)} files ({_elapsed:.0f}s elapsed)", flush=True)
        song_title = midi_file.split('/')[-2]

        if song_title in eval_set:
            mode = 'eval'
        elif song_title in test_set:
            mode = 'test'
        elif split_json and song_title not in train_set:
            # SSoT mode: file not in any bucket of split.json — skip.
            # Happens for files in EXCLUDED_FROM_TEST (e.g.
            # 319_Miles_Davis_Miles_Runs_the_Voodoo_Down_Solo).
            continue
        else:
            mode = 'train'
        os.makedirs(os.path.join(dir_name, mode, song_title), exist_ok=True)
        key_count = len(sorted(glob.glob(os.path.join(dir_name, mode, song_title, '*_+0_*.pkl')))) # in case of modulation

        if shift:
            pitch_shift = range(-5, 7)
        else:
            pitch_shift = [0]
        for k in pitch_shift:
            instances = extract_instances_from_midi(
                midi_path=midi_file,
                num_bars=num_bars,
                frame_per_bar=frame_per_bar,
                stride_bars=stride_bars,
                pitch_range=pitch_range,
                k_shift=k,
                beat_per_bar=beat_per_bar,
                bpm=bpm,
            )
            ps = ('%d' % k) if (k < 0) else ('+%d' % k)
            for inst in instances:
                pkl_filename = os.path.join(
                    dir_name, mode, song_title,
                    '%s_%02d_%s_%02d.pkl' % (song_title, key_count, ps, inst['window_idx'])
                )
                # Drop wrapper-only fields (window_idx, base_note) before pickling so
                # the on-disk pkl format remains identical to the pre-refactor version.
                payload = {
                    'pitch': inst['pitch'],
                    'rhythm': inst['rhythm'],
                    'chord': inst['chord'],
                }
                with open(pkl_filename, 'wb') as f:
                    pickle.dump(payload, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/data2/score2midi')
    parser.add_argument('--midi_dir', type=str, default='cleansed_midi_twotrack_ckey')
    parser.add_argument('--num_bars', type=int, default=8)
    parser.add_argument('--stride_bars', type=int, default=None,
                        help='Sliding-window stride in bars. Default: num_bars // 2 (50%% overlap).')
    parser.add_argument('--frame_per_bar', type=int, default=16)
    parser.add_argument('--pitch_range', type=int, default=48)
    parser.add_argument('--shift', dest='shift', action='store_true')
    parser.add_argument('--split-json', type=str, default=None,
                        help='Path to diploma2/pipelines/training-pipeline/wjazzd_split.json. '
                             'When set, bucket assignments come from the cross-model SSoT '
                             '(EXCLUDED_FROM_TEST applied). When omitted, falls back to the '
                             'legacy random.seed(0) sample (bit-exact identical for train/eval).')

    args = parser.parse_args()
    root_dir = args.root_dir
    midi_dir = args.midi_dir
    num_bars = args.num_bars
    stride_bars = args.stride_bars if args.stride_bars is not None else num_bars // 2
    frame_per_bar = args.frame_per_bar
    pitch_range = args.pitch_range
    shift = args.shift
    split_json = args.split_json

    make_instance_pkl_files(root_dir, midi_dir, num_bars, frame_per_bar, stride_bars,
                            pitch_range, shift, split_json=split_json)
