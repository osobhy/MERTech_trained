# extract_guzheng.py
import os, csv, shutil
from datasets import load_dataset, Audio
import soundfile as sf

DATASET = "ccmusic-database/Guzheng_Tech99"
SPLITS  = ["train", "validation", "test"]
OUTDIR  = "/workspace/MERTech/data/Guzheng_Tech99"

for split in SPLITS:
    ds = load_dataset(DATASET, name="default", split=split)
    ds = ds.cast_column("audio", Audio(decode=True))

    wav_dir = os.path.join(OUTDIR, "audio", split)
    lbl_dir = os.path.join(OUTDIR, "labels", split)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Prepare CSV
    csv_path = os.path.join(lbl_dir, f"{split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "onset_time", "offset_time", "IPT", "note"])
        for i, ex in enumerate(ds):
            fn = f"{split}_{i:05d}.wav"
            wav_path = os.path.join(wav_dir, fn)
            sf.write(wav_path, ex["audio"]["array"], ex["audio"]["sampling_rate"])
            writer.writerow([fn,
                             ex["onset_time"],
                             ex["offset_time"],
                             ex["IPT"],
                             ex["note"]])
