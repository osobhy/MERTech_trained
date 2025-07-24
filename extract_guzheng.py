import os
import csv
from datasets import load_dataset, Audio

ROOT = "/workspace/Guzheng_Tech99"  # change if needed
SPLITS = ["train", "validation", "test"]

for split in SPLITS:
    # Load split (without decoding audio yet)
    ds = load_dataset("ccmusic-database/Guzheng_Tech99", split=split, cache_dir=ROOT)
    ds = ds.cast_column("audio", Audio(decode=True))  # now decode audio

    wav_dir  = f"{ROOT}/data/{split}"
    csv_path = f"{ROOT}/labels/{split}.csv"

    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["onset_time", "offset_time", "note", "IPT"])

        for i, ex in enumerate(ds):
            wav_path = os.path.join(wav_dir, f"{i}.wav")
            ex["audio"]["array"].tofile(wav_path)
            writer.writerow([ex["onset_time"], ex["offset_time"], ex["note"], ex["IPT"]])
