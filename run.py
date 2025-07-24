import datetime

date = datetime.datetime.now()

import sys
sys.path.append('./function')   # ensure our modules are on PYTHONPATH

import os
import random
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, DataCollatorWithPadding
from datasets import load_dataset, Audio

from function.fit import *
from function.model import *
from function.config import *  # provides URL, BATCH_SIZE, MERT_SAMPLE_RATE, NUM_LABELS, MAX_MIDI, MIN_MIDI, FREEZE_ALL, saveName

# Reproducibility

def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

get_random_seed(42)

# Output model directory
out_model_fn = f"./data/model/{date.year}{date.month}{date.day}{date.hour}:{date.minute}:{date.second}/{saveName}/"
os.makedirs(out_model_fn, exist_ok=True)

# Utility to compute class-imbalance weights

def get_weight(Ytr):  # expects shape (num_samples, num_classes, num_frames)
    mp = Ytr.sum(0).sum(0)            # (num_classes,)
    mmp = mp.float() / mp.sum()
    cc = ((mmp.mean() / mmp) * ((1 - mmp) / (1 - mmp.mean()))) ** 0.3
    return cc

# ————————————————————————————————
# 1) Load & preprocess via Hugging Face Datasets
# ————————————————————————————————
processor = Wav2Vec2FeatureExtractor.from_pretrained(URL, trust_remote_code=True)

# Placeholder for label-generation logic
from function.load_data import make_label_matrix  # implement this to mirror your old loader

def build_labels_matrix(example):
    """
    Turn example's onset_time, offset_time, IPT, and note arrays
    into a (num_frames × num_classes) torch.Tensor of labels.
    """
    return make_label_matrix(
        example['onset_time'],
        example['offset_time'],
        example['IPT'],
        example['note'],
        sr=MERT_SAMPLE_RATE
    )

# Load splits

def prepare_split(split_name):
    ds = load_dataset("ccmusic-database/Guzheng_Tech99",
                      split=split_name,
                      cache_dir="./hf_cache")
    ds = ds.cast_column("audio", Audio(decode=True))

    def map_fn(ex):
        inputs = processor(
            ex["audio"]["array"],
            sampling_rate=MERT_SAMPLE_RATE,
            return_tensors="pt",
            padding=False
        )["input_values"].squeeze(0)

        lab = ex["label"]
        labels = build_labels_matrix(
            lab["onset"],
            lab["offset"],
            lab["technique"],
            lab["pitch"],
            total_samples=len(ex["audio"]["array"])
        )
        return {"input_values": inputs, "labels": labels}

    return ds.map(
        map_fn,
        remove_columns=["audio", "mel", "label"],
        batched=False
    )

train_ds = prepare_split('train')
val_ds   = prepare_split('validation')
print('✔ Loaded HF dataset splits.')

# Collator for padding
collator = DataCollatorWithPadding(processor=processor, return_tensors='pt')

# PyTorch DataLoaders
tr_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    collate_fn=collator
)
va_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=1,
    num_workers=2,
    collate_fn=collator
)
print('✔ DataLoaders are ready.')

# Compute class-imbalance weights from all train labels
all_labels = torch.stack([ex['labels'] for ex in train_ds], dim=0)  # (N, T, C)
# transpose to (N, C, T)
inverse_freq = get_weight(all_labels.transpose(0,2,1))

# ————————————————————————————————
# 2) Build model and train
# ————————————————————————————————
model = SSLNet(
    url=URL,
    class_num=NUM_LABELS*(MAX_MIDI-MIN_MIDI+1),
    weight_sum=1,
    freeze_all=FREEZE_ALL
).to(device)

print('Model structure:')
print(Visualization(model).structure_graph())

trainer = Trainer(
    model,
    lr=1e-3,
    max_steps=10000,
    output_dir=out_model_fn,
    validation_interval=5,
    save_interval=100
)
trainer.fit(tr_loader, va_loader, inverse_freq)
