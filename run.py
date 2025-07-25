import datetime
import sys
import os
import random
import argparse

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

# keep original relative‑imports exactly as before
sys.path.append('./function')
from function.fit import *
from function.model import *
from function.load_data import *
from function.config  import *

# ──────────────────────────────── tiny CLI patch ─────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--resume_ckpt', type=str, default=None,
                    help='Path to *.pth checkpoint to resume from')
parser.add_argument('--extra_epochs', type=int, default=0,
                    help='Train this many *additional* epochs')
args = parser.parse_args()
# ───────────────────────────────── device printout ───────────────────────────
print("Using device:", device)
print(" torch.cuda.is_available():", torch.cuda.is_available())
print(" torch.cuda.device_count():", torch.cuda.device_count())
if device.type == "cuda":
    print(" current_device:", torch.cuda.current_device())
    print(" device_name   :", torch.cuda.get_device_name(device.index))

max_epochs = int(os.getenv("MAX_EPOCHS", "10000"))  # untouched

# ───────────────────────── reproducibility (original code) ───────────────────

def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
get_random_seed(42)

# ───────────── helper for weighted loss (same as original) ───────────────────

def get_weight(Ytr):  # (N, T, C)
    mp  = Ytr[:].sum(0).sum(0)
    mmp = mp.astype(np.float32) / mp.sum()
    cc  = ((mmp.mean() / mmp) * ((1 - mmp) / (1 - mmp.mean()))) ** 0.3
    return torch.from_numpy(cc)

# ───────────────────────────── output folder (unchanged) ─────────────────────
now           = datetime.datetime.now()
out_model_fn  = f"./data/model/{now.year}{now.month}{now.day}{now.hour}:{now.minute}:{now.second}/{saveName}/"
if not os.path.exists(out_model_fn):
    os.makedirs(out_model_fn)

# ─────────────────────────────── data loading (as is) ────────────────────────
wav_dir = DATASET + '/data'
csv_dir = DATASET + '/labels'

Xtr, Ytr, Ytr_p, Ytr_o, avg, std        = load(wav_dir, csv_dir, ['train'])
Xva, Yva, Yva_p, Yva_o, va_avg, va_std = load(wav_dir, csv_dir, ['validation'], avg, std)
print('finishing data loading...')

processor = Wav2Vec2FeatureExtractor.from_pretrained(URL, trust_remote_code=True)
Xtrs = processor(Xtr, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")
Xvas = processor(Xva, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")

# Build Dataloader (identical)
tr_loader = torch.utils.data.DataLoader(
    Data2Torch2([Xtrs["input_values"], Ytr, Ytr_p, Ytr_o]),
    shuffle=True, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, drop_last=True)
va_loader = torch.utils.data.DataLoader(
    Data2Torch2([Xvas["input_values"], Yva, Yva_p, Yva_o]),
    batch_size=1, num_workers=2, pin_memory=True)
print('finishing data building...')

# ───────────────────────────── model construction ────────────────────────────
model = SSLNet(url=URL, class_num=NUM_LABELS*(MAX_MIDI-MIN_MIDI+1),
               weight_sum=1, freeze_all=FREEZE_ALL).to(device)

# ──────────────────────── minimal resume‑from‑ckpt patch ─────────────────────
start_epoch = 0  # just for logging; Trainer will still run 0‑based epochs
if args.resume_ckpt is not None:
    print(f"Loading checkpoint: {args.resume_ckpt}")
    ckpt = torch.load(args.resume_ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    start_epoch = ckpt.get('epoch', 0) + 1
    print(f"✔ Resumed weights from epoch {start_epoch - 1}")

# ───────────────── loss‑weight & trainer call (almost unchanged) ─────────────
inverse_feq = get_weight(Ytr.transpose(0, 2, 1))

total_epochs = 200 + args.extra_epochs  # extend schedule if requested

Trer = Trainer(model, 1e-3, total_epochs, out_model_fn,
               validation_interval=5, save_interval=100)

Trer.fit(tr_loader, va_loader, inverse_feq)
