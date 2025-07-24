#!/usr/bin/env python3
import os, io, csv
from datasets import load_dataset, Audio
import soundfile as sf

# Where to dump the data
ROOT = "/workspace/MERTech/data/Guzheng_Tech99"
SPLITS = ["train", "validation", "test"]

for split in SPLITS:
    print(f"→ Processing split '{split}'")
    # 1) Load the Arrow for this split
    ds = load_dataset(
        "ccmusic-database/Guzheng_Tech99",
        split=split,
        cache_dir=ROOT + "/hf_cache"
    )
    # 2) Get raw bytes of each audio file (no torchcodec)
    ds = ds.cast_column("audio", Audio(decode=False))

    # 3) Prepare output folders
    wav_dir = os.path.join(ROOT, "audio", split)
    csv_dir = os.path.join(ROOT, "labels", split)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # 4) Iterate examples
    writer = None
    for i, ex in enumerate(ds):
        fn = f"{split}_{i:05d}"
        wav_path = os.path.join(wav_dir, fn + ".wav")
        csv_path = os.path.join(csv_dir, fn + ".csv")

        # Decode from raw bytes
        audio_bytes = ex["audio"]["bytes"]
        if audio_bytes is None:
            raise RuntimeError(f"No embedded bytes for example {i} in split {split}")
        arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # Write out WAV
        sf.write(wav_path, arr, sr)

        # Write out CSV (one row per note event)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["onset_time", "offset_time", "IPT", "note"])
            for onset, offset, ipt, note in zip(
                ex["onset_time"],
                ex["offset_time"],
                ex["IPT"],
                ex["note"],
            ):
                w.writerow([onset, offset, ipt, note])

    print(f"  • Wrote {len(ds)} WAV+CSV pairs to {split}")

print("✅ Extraction complete.")
