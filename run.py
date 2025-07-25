import datetime
import sys
import os
import random
import argparse  # NEW

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

sys.path.append('./function')
from function.fit import *
from function.model import *
from function.load_data import *
from function.config import *

# Parse optional resume / extra‑epoch flags
parser = argparse.ArgumentParser()  # NEW
parser.add_argument('--resume_ckpt', type=str, default=None, help='checkpoint path')  # NEW
parser.add_argument('--extra_epochs', type=int, default=0, help='additional epochs')  # NEW
args = parser.parse_args()  # NEW

print("Using device:", device)
print(" torch.cuda.is_available():", torch.cuda.is_available())
print(" torch.cuda.device_count():", torch.cuda.device_count())
if device.type == "cuda":
    print(" current_device:", torch.cuda.current_device())
    print(" device_name   :", torch.cuda.get_device_name(device.index))
max_epochs = int(os.getenv("MAX_EPOCHS", "10000"))

def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
get_random_seed(42)

#Obtain the weight of the loss function to alleviate class imbalance
def get_weight(Ytr):#(1508, 375, 7)
    mp = Ytr[:].sum(0).sum(0) #(7,)
    mmp = mp.astype(np.float32) / mp.sum()
    cc=((mmp.mean() / mmp) * ((1-mmp)/(1 - mmp.mean())))**0.3
    inverse_feq = torch.from_numpy(cc)
    return inverse_feq

date = datetime.datetime.now()  # keep after seed
out_model_fn = './data/model/%d%d%d%d:%d:%d/%s/'%(date.year,date.month,date.day,date.hour,date.minute,date.second,saveName)
if not os.path.exists(out_model_fn):
    os.makedirs(out_model_fn)

# load data
wav_dir = DATASET + '/data'
csv_dir = DATASET + '/labels'

groups = ['train']
vali_groups = ['validation']

Xtr,Ytr,Ytr_p,Ytr_o,avg,std = load(wav_dir,csv_dir,groups)
Xva,Yva,Yva_p,Yva_o,va_avg,va_std = load(wav_dir,csv_dir,vali_groups,avg,std)
print ('finishing data loading...')

processor = Wav2Vec2FeatureExtractor.from_pretrained(URL, trust_remote_code=True)
Xtrs = processor(Xtr, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")
Xvas = processor(Xva, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")

# Build Dataloader
t_kwargs = {'batch_size': BATCH_SIZE, 'num_workers': 2, 'pin_memory': True, 'drop_last': True}
v_kwargs = {'batch_size': 1, 'num_workers': 2, 'pin_memory': True}
tr_loader = torch.utils.data.DataLoader(Data2Torch2([Xtrs["input_values"], Ytr, Ytr_p, Ytr_o]), shuffle=True, **t_kwargs)
va_loader = torch.utils.data.DataLoader(Data2Torch2([Xvas["input_values"], Yva, Yva_p, Yva_o]), **v_kwargs)
print ('finishing data building...')

model = SSLNet(url=URL, class_num=NUM_LABELS*(MAX_MIDI-MIN_MIDI+1),weight_sum=1,freeze_all=FREEZE_ALL).to(device)

# Resume weights if provided
if args.resume_ckpt:  # NEW
    ckpt = torch.load(args.resume_ckpt, map_location='cpu')  # NEW
    model.load_state_dict(ckpt['model'])  # NEW
    print(f"Resumed from {args.resume_ckpt}")  # NEW

inverse_feq = get_weight(Ytr.transpose(0,2,1))

total_epochs = 200 + args.extra_epochs  # NEW

Trer = Trainer(model, 1e-3, total_epochs, out_model_fn, validation_interval=5, save_interval=100)
Trer.fit(tr_loader, va_loader,inverse_feq)
