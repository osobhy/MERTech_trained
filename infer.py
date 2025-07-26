import argparse
import sys
import numpy as np
import torch
import librosa
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor

sys.path.append('./function')
from config import *
from model import SSLNet


def get_wav(path: str) -> np.ndarray:
    """Load audio file and resample to MERT_SAMPLE_RATE."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    if sr != MERT_SAMPLE_RATE:
        resampler = T.Resample(sr, MERT_SAMPLE_RATE)
        y = resampler(torch.from_numpy(y)).numpy()
    return y


def chunk_wav_test(wav: np.ndarray):
    """Split waveform into TIME_LENGTH-long segments."""
    s = int(MERT_SAMPLE_RATE * TIME_LENGTH)
    length = int(np.ceil(len(wav) / s) * s)
    padded = np.concatenate((wav, np.zeros(length - len(wav))), 0)
    return [padded[i * s:(i + 1) * s] for i in range(length // s)]


def merge_segments(pred: np.ndarray, threshold: float):
    """Convert a probability sequence to (start, end) segments."""
    detections = []
    start = None
    for i, p in enumerate(pred):
        if p >= threshold and start is None:
            start = i / FEATURE_RATE
        elif p < threshold and start is not None:
            detections.append((start, i / FEATURE_RATE))
            start = None
    if start is not None:
        detections.append((start, len(pred) / FEATURE_RATE))
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect tremolo techniques in an audio file")
    parser.add_argument("--ckpt", required=True, help="Path to SSLNet checkpoint")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for tremolo detection")
    parser.add_argument("--save", metavar="PATH", help="Optional path to save full prediction array")
    args = parser.parse_args()

    model = SSLNet(url=URL, class_num=NUM_LABELS*(MAX_MIDI-MIN_MIDI+1), weight_sum=1, freeze_all=FREEZE_ALL).to(device)
    state_dict = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    processor = Wav2Vec2FeatureExtractor.from_pretrained(URL, trust_remote_code=True)

    wav = get_wav(args.audio)
    chunks = chunk_wav_test(wav)

    tremolo_probs = []
    with torch.no_grad():
        for chunk in chunks:
            data = processor(chunk, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")["input_values"].float().to(device)
            IPT_pred, _, _ = model(data)
            probs = torch.sigmoid(IPT_pred.squeeze(0)).cpu().numpy()
            tremolo_probs.append(probs[5])  # index 5 corresponds to tremolo

    tremolo_probs = np.concatenate(tremolo_probs)
    if args.save:
        np.save(args.save, tremolo_probs)

    detections = merge_segments(tremolo_probs, args.threshold)
    if detections:
        print("Tremolo segments (seconds):")
        for s, e in detections:
            print(f"{s:.2f} - {e:.2f}")
    else:
        print("No tremolo detected.")


if __name__ == "__main__":
    main()