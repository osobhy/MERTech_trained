import argparse
import os
import sys
import numpy as np
import torch
import librosa
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor

sys.path.append('./function')
from config import *
from model import SSLNet


def get_wav(file):
    """Load audio and resample to MERT_SAMPLE_RATE"""
    sr = SAMPLE_RATE
    y, sr = librosa.load(file, sr=sr)
    sampling_rate = sr
    resample_rate = MERT_SAMPLE_RATE
    if resample_rate != sampling_rate:
        resampler = T.Resample(sampling_rate, resample_rate)
        input_audio = resampler(torch.from_numpy(y)).numpy()
    else:
        input_audio = y
    return input_audio


def chunk_wav_test(f):
    """Split waveform into TIME_LENGTH segments"""
    s = int(MERT_SAMPLE_RATE * TIME_LENGTH)
    xdata = f
    x = []
    length = int(np.ceil(len(xdata) / s) * s)
    app = np.zeros((length - xdata.shape[0]))
    xdata = np.concatenate((xdata, app), 0)
    for i in range(int(length / s)):
        data = xdata[int(i * s):int(i * s + s)]
        x.append(data)
    return x


def main():
    parser = argparse.ArgumentParser(description="Run inference on an audio file")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--out", default="prediction.npy", help="File to save numpy predictions")
    args = parser.parse_args()

    model = SSLNet(url=URL, class_num=NUM_LABELS*(MAX_MIDI-MIN_MIDI+1), weight_sum=1, freeze_all=FREEZE_ALL).to(device)
    state_dict = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    processor = Wav2Vec2FeatureExtractor.from_pretrained(URL, trust_remote_code=True)

    audio = get_wav(args.audio)
    chunks = chunk_wav_test(audio)

    all_preds = []
    with torch.no_grad():
        for chunk in chunks:
            data = processor(chunk, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")["input_values"].float().to(device)
            IPT_pred, pitch_pred, onset_pred = model(data)
            probs = torch.sigmoid(IPT_pred.squeeze(0)).cpu().numpy()
            all_preds.append(probs)

    prediction = np.concatenate(all_preds, axis=-1)
    np.save(args.out, prediction)
    print(f"Saved predictions to {args.out}. Shape: {prediction.shape}")


if __name__ == "__main__":
    main()