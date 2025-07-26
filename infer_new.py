#!/usr/bin/env python
"""
Infer tremolo spans on a single WAV file.

Usage:
    python tools/infer_one.py \
           --ckpt checkpoints/oud_tremolo_best.pt \
           --wav  path/to/your_oud_take.wav \
           --out_csv tremolo_segments.csv
"""
import argparse, torch, torchaudio
from transformers import Wav2Vec2FeatureExtractor
sys.path.append('./function')

# ────────────────────────────────────────────────────────────────────────────
# Repo‑level imports (grab all the config constants you already have)
# ────────────────────────────────────────────────────────────────────────────
from function.model import SSLNet
from function.config import (URL, device, FEATURE_RATE,
                             MIN_MIDI, MAX_MIDI, NUM_LABELS,
                             MERT_SAMPLE_RATE, FREEZE_ALL)

# ────────────────────────────────────────────────────────────────────────────
# Tunables that depend on your label set
# ────────────────────────────────────────────────────────────────────────────
TREMOLO_CLASS_ID = 5            # 0‑based index in your IPT list
FRAME_THR        = 0.50         # probability threshold for “tremolo”

# Derived constants
HOP_SEC   = 1.0 / FEATURE_RATE                    # 75 fps → 13.33 ms
MIDI_BINS = MAX_MIDI - MIN_MIDI + 1               # 52 bins (36…87)
TREMOLO_SLICE = slice(TREMOLO_CLASS_ID * MIDI_BINS,
                      (TREMOLO_CLASS_ID + 1) * MIDI_BINS)

# ────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str) -> SSLNet:
    model = SSLNet(url=URL,
                   class_num=NUM_LABELS * MIDI_BINS,
                   weight_sum=1,
                   freeze_all=FREEZE_ALL).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model

def forward(model: SSLNet, wav_path: str):
    # load & resample to 24 kHz mono
    wav, sr = torchaudio.load(wav_path)
    if sr != MERT_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, MERT_SAMPLE_RATE)
    wav = wav.mean(0, keepdim=True)               # ensure mono
    # MERT feature extractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained(URL,
                                                         trust_remote_code=True)
    x = processor(wav.squeeze(0),
                  sampling_rate=MERT_SAMPLE_RATE,
                  return_tensors="pt")["input_values"].to(device)
    with torch.no_grad():
        IPT_logits, _, _ = model(x.float())
    # sigmoid → probability, then pick tremolo slice and max‑pool over pitch
    proba = IPT_logits.sigmoid()[0]               # [NUM_LABELS*MIDI_BINS, T]
    tremolo_proba = proba[TREMOLO_SLICE].max(0).values.cpu().numpy()
    return tremolo_proba                          # shape: [T]

def proba_to_segments(pred, thr=FRAME_THR, hop_sec=HOP_SEC):
    """Convert per‑frame probability to (start, end) time pairs in seconds."""
    active = (pred > thr).astype(int)
    segs, start = [], None
    for i, v in enumerate(active):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start * hop_sec, i * hop_sec))
            start = None
    # handle trailing segment
    if start is not None:
        segs.append((start * hop_sec, len(active) * hop_sec))
    return segs

# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",     required=True, help="Path to .pt checkpoint")
    ap.add_argument("--wav",      required=True, help="24 kHz (or any) WAV file")
    ap.add_argument("--out_csv",  default="tremolo_segments.csv",
                                    help="Destination CSV")
    args = ap.parse_args()

    model = load_model(args.ckpt)
    tremolo_proba = forward(model, args.wav)
    segments = proba_to_segments(tremolo_proba)

    # save CSV
    with open(args.out_csv, "w") as f:
        f.write("start_s,end_s,label\n")
        for s, e in segments:
            f.write(f"{s:.3f},{e:.3f},tremolo\n")

    print(f"Detected {len(segments)} tremolo spans → {args.out_csv}")
