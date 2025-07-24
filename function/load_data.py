import os
import numpy as np
import torch
from glob import glob
import csv
from datasets import load_dataset, Audio
import torchaudio.transforms as T
from config import *  # expects SAMPLE_RATE, MERT_SAMPLE_RATE, FEATURE_RATE, TIME_LENGTH, HOPS_IN_ONSET, NUM_LABELS, MAX_MIDI, MIN_MIDI

# ----------------------------------------------------------------------------
# This loader pulls from HuggingFace directly, bypassing on-disk WAV/CSV files.
# It mirrors the original batch/chunking logic.
# ----------------------------------------------------------------------------

def make_label_matrix(onset_times, offset_times, IPTs, notes, total_samples):
    """
    Build (num_frames x num_classes) label tensor for one audio example.
    - onset_times, offset_times: lists of floats (seconds)
    - IPTs: list of label strings
    - notes: list of MIDI numbers
    - total_samples: int, length of audio array in samples at MERT_SAMPLE_RATE
    """
    # Compute number of feature frames at FEATURE_RATE
    num_frames = int(np.ceil((total_samples / MERT_SAMPLE_RATE) * FEATURE_RATE))
    n_IPTs  = NUM_LABELS
    n_keys  = MAX_MIDI - MIN_MIDI + 1

    IPT_label   = np.zeros((n_IPTs, num_frames), dtype=int)
    pitch_label = np.zeros((n_keys, num_frames), dtype=int)
    onset_label = np.zeros((1, num_frames), dtype=int)

    # map from IPT string to index
    technique_map = {
        'chanyin': 0, 'boxian': 1, 'shanghua': 2, 'xiahua': 3,
        'huazhi': 4, 'lianmo': 4, 'liantuo': 4, 'guazou': 4,
        'yaozhi': 5, 'dianyin': 6
    }

    for onset, offset, ipt, note in zip(onset_times, offset_times, IPTs, notes):
        left_frame = int(np.round(onset * FEATURE_RATE))
        right_frame = int(np.round(offset * FEATURE_RATE))
        right_frame = min(right_frame, num_frames)
        ipt_idx = technique_map[ipt]
        note_idx = int(note) - MIN_MIDI
        # mark labels
        IPT_label[ipt_idx, left_frame:right_frame] = 1
        pitch_label[note_idx, left_frame:right_frame] = 1
        # onset window
        onset_end = min(left_frame + HOPS_IN_ONSET, num_frames)
        onset_label[0, left_frame:onset_end] = 1

    # stack into shape (frames, classes_total)
    # total classes = IPTs + pitches + onset
    labels = np.concatenate([
        IPT_label,
        pitch_label,
        onset_label
    ], axis=0)  # shape (C, T)
    labels = torch.from_numpy(labels.T).long()  # (T, C)
    return labels

# ----------------------------------------------------------------------------
# Main loader: mirrors signature load(wav_dir, csv_dir, groups)
# 'wav_dir' should be the HF dataset name, e.g. 'ccmusic-database/Guzheng_Tech99'
# 'csv_dir' is ignored in this mode.
# 'groups' is a list of splits: ['train'], ['validation'], ['test']
# ----------------------------------------------------------------------------

def load(wav_dir, csv_dir, groups, avg=None, std=None):
    # wav_dir is actually the dataset identifier
    ds = load_dataset(wav_dir, cache_dir='./hf_cache')

    # cast audio column to raw arrays
    ds = ds.cast_column('audio', Audio(decode=True))

    data = {}
    for split in groups:
        split_ds = ds[split]
        features = []
        labels_list = []
        for ex in split_ds:
            # 1) get raw audio and resample if needed
            audio_arr = ex['audio']['array']
            sr = ex['audio']['sampling_rate']
            if sr != MERT_SAMPLE_RATE:
                resampler = T.Resample(sr, MERT_SAMPLE_RATE)
                audio_arr = resampler(torch.from_numpy(audio_arr)).numpy()
            total_samples = len(audio_arr)
            # 2) build label matrix
            onset_times  = ex['onset_time']
            offset_times = ex['offset_time']
            IPTs         = ex['IPT']
            notes        = ex['note']
            lab = make_label_matrix(onset_times, offset_times, IPTs, notes, total_samples)

            features.append(torch.from_numpy(audio_arr).float())
            labels_list.append(lab)

        data[split] = (features, labels_list)

    # now return in same order as original: Xtr,Ytr,Ytr_p,Ytr_o,avg,std for train,
    # but since we're streaming, you can adapt run.py to accept features + labels directly.
    return data
