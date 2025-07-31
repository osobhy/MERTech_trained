import datetime
import sys
import os
import random

import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor

# our lib
sys.path.append('./function')
from function.fit import Trainer
from function.model import SSLNet
from function.load_data import load, Data2Torch2
from function.config import *

# ------------------------------------------------------------------------------
# Utility to freeze / unfreeze the MERT encoder
# ------------------------------------------------------------------------------
def freeze_encoder(model):
    for name, param in model.named_parameters():
        # adjust this substring to match your encoder's module names
        if 'wav2vec2' in name or 'feature_extractor' in name:
            param.requires_grad = False

def unfreeze_encoder(model):
    for param in model.parameters():
        param.requires_grad = True

# ------------------------------------------------------------------------------
# Boilerplate & seeding
# ------------------------------------------------------------------------------
date = datetime.datetime.now()
print("Using device:", device)
print(" torch.cuda.is_available():", torch.cuda.is_available())
print(" torch.cuda.device_count():", torch.cuda.device_count())
if device.type == "cuda":
    print(" current_device:", torch.cuda.current_device())
    print(" device_name   :", torch.cuda.get_device_name(device.index))

max_epochs = int(os.getenv("MAX_EPOCHS", str(max_epochs if 'max_epochs' in locals() else 2000)))
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)

# ------------------------------------------------------------------------------
# Prepare output directory
# ------------------------------------------------------------------------------
out_model_fn = f'./data/model/{date.year}{date.month:02d}{date.day:02d}{date.hour:02d}{date.minute:02d}{date.second:02d}/{saveName}/'
os.makedirs(out_model_fn, exist_ok=True)

# ------------------------------------------------------------------------------
# Load & preprocess data
# ------------------------------------------------------------------------------
wav_dir = os.path.join(DATASET, 'data')
csv_dir = os.path.join(DATASET, 'labels')

Xtr, Ytr, Ytr_p, Ytr_o, avg, std = load(wav_dir, csv_dir, ['train'])
Xva, Yva, Yva_p, Yva_o, _, _       = load(wav_dir, csv_dir, ['validation'], avg, std)
print('✔️  Finished data loading')

processor = Wav2Vec2FeatureExtractor.from_pretrained(URL, trust_remote_code=True)
Xtrs = processor(Xtr, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")
Xvas = processor(Xva, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")

t_kwargs = {'batch_size': BATCH_SIZE, 'num_workers': 21, 'pin_memory': True, 'drop_last': True}
v_kwargs = {'batch_size': 1,            'num_workers': 2,  'pin_memory': True}
tr_loader = torch.utils.data.DataLoader(
    Data2Torch2([Xtrs["input_values"], Ytr, Ytr_p, Ytr_o]),
    shuffle=True, **t_kwargs
)
va_loader = torch.utils.data.DataLoader(
    Data2Torch2([Xvas["input_values"], Yva, Yva_p, Yva_o]),
    **v_kwargs
)
print('✔️  Finished building DataLoaders')

# ------------------------------------------------------------------------------
# Build model
# ------------------------------------------------------------------------------
model = SSLNet(
    url=URL,
    class_num=NUM_LABELS * (MAX_MIDI - MIN_MIDI + 1),
    weight_sum=True,     # ← learnable weighted sum of all 13 MERT layers
    freeze_all=False     # we'll control freezing manually below
).to(device)

# ------------------------------------------------------------------------------
# Class-imbalance weighting
# ------------------------------------------------------------------------------
def get_weight(Y):
    # Y has shape (N_samples, T_frames, NUM_LABELS)
    mp  = Y.sum(0).sum(0).astype(np.float32)                # (NUM_LABELS,)
    mmp = mp / mp.sum()
    cc  = ((mmp.mean()/mmp) * ((1-mmp)/(1-mmp.mean())))**0.3
    return torch.from_numpy(cc)

inverse_feq = get_weight(Ytr.transpose(0,2,1))
print('✔️  Computed class-balance weights')

# ------------------------------------------------------------------------------
# Two-step finetuning via Trainer
# ------------------------------------------------------------------------------
# Trainer signature is Trainer(model, lr, epochs, out_dir, validation_interval, save_interval)
base_lr = 1e-3

if TWO_STEP:
    # ─── Phase 1: warm up only the heads ───────────────────────────────────────
    print(f"\n=== Phase 1: Training heads only for {LIN_EPOCH} epochs ===")
    freeze_encoder(model)
    trainer1 = Trainer(model, base_lr, LIN_EPOCH, out_model_fn,
                       validation_interval=5, save_interval=100)
    trainer1.fit(tr_loader, va_loader, inverse_feq)

    # reload best checkpoint from phase1
    best1 = os.path.join(out_model_fn, 'best_model.pt')
    model.load_state_dict(torch.load(best1, map_location=device))

    # ─── Phase 2: unfreeze full model ──────────────────────────────────────────
    print(f"\n=== Phase 2: Unfreezing encoder, training full model for {max_epochs-LIN_EPOCH} more epochs ===")
    unfreeze_encoder(model)
    trainer2 = Trainer(model, base_lr, max_epochs - LIN_EPOCH, out_model_fn,
                       validation_interval=5, save_interval=100)
    trainer2.fit(tr_loader, va_loader, inverse_feq)

else:
    # Single-stage finetuning
    print(f"\n=== Single-stage finetuning for {max_epochs} epochs ===")
    trainer = Trainer(model, base_lr, max_epochs, out_model_fn,
                      validation_interval=5, save_interval=100)
    trainer.fit(tr_loader, va_loader, inverse_feq)
