import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect tremolo techniques in an audio file")
    parser.add_argument("--ckpt", required=True,
                        help="Path to SSLNet checkpoint")
    parser.add_argument("--audio", required=True,
                        help="Path to input audio file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for tremolo detection")
    parser.add_argument("--save", metavar="PATH",
                        help="Optional path to save full tremolo probability array")
    parser.add_argument("--post", action="store_true",
                        help="Apply onset-based post-processing")
    args = parser.parse_args()

    import numpy as np
    import torch
    import librosa
    import torchaudio.transforms as T
    from transformers import Wav2Vec2FeatureExtractor

    sys.path.append('./function')
    import config
    from model import SSLNet
    from lib import extract_notes

    def get_wav(path: str) -> np.ndarray:
        """Load audio file and resample to MERT_SAMPLE_RATE."""
        y, sr = librosa.load(path, sr=config.SAMPLE_RATE)
        if sr != config.MERT_SAMPLE_RATE:
            resampler = T.Resample(sr, config.MERT_SAMPLE_RATE)
            y = resampler(torch.from_numpy(y)).numpy()
        return y

    def chunk_wav_test(wav: np.ndarray):
        """Split waveform into TIME_LENGTH-long segments."""
        s = int(config.MERT_SAMPLE_RATE * config.TIME_LENGTH)
        length = int(np.ceil(len(wav) / s) * s)
        padded = np.concatenate((wav, np.zeros(length - len(wav))), 0)
        return [padded[i * s:(i + 1) * s] for i in range(length // s)]

    def merge_segments(pred: np.ndarray, threshold: float):
        """Convert a probability sequence to (start, end) segments."""
        detections = []
        start = None
        for i, p in enumerate(pred):
            if p >= threshold and start is None:
                start = i / config.FEATURE_RATE
            elif p < threshold and start is not None:
                detections.append((start, i / config.FEATURE_RATE))
                start = None
        if start is not None:
            detections.append((start, len(pred) / config.FEATURE_RATE))
        return detections

    wav = get_wav(args.audio)
    chunks = chunk_wav_test(wav)

    model = SSLNet(url=config.URL,
                   class_num=config.NUM_LABELS * (config.MAX_MIDI - config.MIN_MIDI + 1),
                   weight_sum=1,
                   freeze_all=config.FREEZE_ALL).to(config.device)
    state_dict = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        config.URL, trust_remote_code=True)

    ipt_probs = []
    onset_probs = []
    with torch.no_grad():
        for chunk in chunks:
            data = processor(
                chunk,
                sampling_rate=config.MERT_SAMPLE_RATE,
                return_tensors="pt")["input_values"].float().to(config.device)
            IPT_pred, _, onset_pred = model(data)
            ipt = torch.sigmoid(IPT_pred.squeeze(0)).cpu().numpy()
            onset = torch.sigmoid(onset_pred.squeeze(0)).cpu().numpy()
            ipt_probs.append(ipt)
            onset_probs.append(onset)

    ipt_probs = np.concatenate(ipt_probs, axis=1)
    onset_probs = np.concatenate(onset_probs, axis=1)

    tremolo_probs = ipt_probs[5]
    if args.save:
        np.save(args.save, tremolo_probs)

    if args.post:
        pitches, intervals = extract_notes(
            torch.from_numpy(onset_probs.T),
            torch.from_numpy(ipt_probs.T),
            onset_threshold=args.threshold,
            frame_threshold=args.threshold)
        detections = [
            (s / config.FEATURE_RATE, e / config.FEATURE_RATE)
            for p, (s, e) in zip(pitches, intervals) if p == 5
        ]
    else:
        detections = merge_segments(tremolo_probs, args.threshold)

    if detections:
        print("Tremolo segments (seconds):")
        for s, e in detections:
            print(f"{s:.2f} - {e:.2f}")
    else:
        print("No tremolo detected.")


if __name__ == "__main__":
    main()
