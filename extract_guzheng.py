#!/usr/bin/env python3
import os, io, csv
from datasets import load_dataset, Audio
import soundfile as sf

# 1) Where MERTech’s run.py will look
BASE = "/workspace/MERTech/data/Guzheng_Tech99"
for split in ["train", "validation", "test"]:
    print(f"→ Processing split '{split}'")

    # 2) Load only metadata (no torchcodec)
    ds = load_dataset(
        "ccmusic-database/Guzheng_Tech99",
        split=split,
        cache_dir=os.path.join(BASE, "hf_cache")
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    # 3) Prepare output dirs that match MERTech’s loader:
    #    DATASET/data/<split>/*.wav  and  DATASET/labels/<split>/*.csv
    wav_out = os.path.join(BASE, "data", split)
    csv_out = os.path.join(BASE, "labels", split)
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(csv_out, exist_ok=True)

    # 4) Iterate and dump
    for i, ex in enumerate(ds):
        fname = f"{split}_{i:05d}"
        wav_bytes = ex["audio"]["bytes"]
        if wav_bytes is None:
            raise RuntimeError(f"No embedded bytes for example {i} in {split}")
        # decode and write WAV
        arr, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        sf.write(os.path.join(wav_out, fname + ".wav"), arr, sr)

        # write one CSV per file, with one row per note
        label = ex["label"]
        with open(os.path.join(csv_out, fname + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["onset_time", "offset_time", "IPT", "note"])
            for o, p, t, n in zip(
                label["onset"],
                label["offset"],
                label["technique"],
                label["pitch"]
            ):
                w.writerow([o, p, t, n])

    print(f"  • Wrote {len(ds)} .wav + .csv pairs for '{split}'")

print("✅ Extraction complete.")
