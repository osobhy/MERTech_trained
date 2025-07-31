import sys
import matplotlib
matplotlib.use('Agg')
import datetime
date = datetime.datetime.now()
sys.path.append('./function')
from function.model import *
from function.lib import *
from function.load_data import *
import numpy as np
import random
import os
import argparse
from transformers import Wav2Vec2FeatureExtractor

#parser = argparse.ArgumentParser()
#parser.add_argument("--ckpt",     required=True)
#args = parser.parse_args()
#ckpt = torch.load(args.ckpt, map_location="cpu")
#print(">>> checkpoint type:", type(ckpt))
#if isinstance(ckpt, dict):
 #   print(">>> ckpt keys:", ckpt.keys())


def start_test():
    def get_random_seed(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    get_random_seed(42)

    #load model
    model = SSLNet(url=URL, class_num=NUM_LABELS*(MAX_MIDI-MIN_MIDI+1), weight_sum=1, freeze_all=FREEZE_ALL).to(device)
    state_dict = torch.load("data/model/202573120:7:59/mul_onset7_pitch_IPT_share_weight_weighted_loss-MERT-v1-95M/best_e_110", map_location="cpu")
    model.load_state_dict(state_dict)

    print('finishing loading model')

    wav_dir = DATASET + '/data'
    csv_dir = DATASET + '/labels'
    test_group = ['test']
    Xte, Yte, Yte_p, Yte_o,  _, _ = load(wav_dir, csv_dir, test_group, None, None)
    print ('finishing loading dataset')

    processor = Wav2Vec2FeatureExtractor.from_pretrained(URL, trust_remote_code=True)

    # start predict
    print('start predicting...')
    model.eval()
    ds = 0
    for i, x in enumerate(Xte):
        data = processor(x, sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")["input_values"].float().to(device)
        target = Yte[i]
        target_p = Yte_p[i]
        target_o = Yte_o[i]
        IPT_pred, pitch_pred, onset_pred = model(data)
        f_pred = F.sigmoid(IPT_pred.squeeze(0)).data.cpu().numpy()
        p_pred = F.sigmoid(pitch_pred.squeeze(0)).data.cpu().numpy()
        o_pred = F.sigmoid(onset_pred.squeeze(0)).data.cpu().numpy()

        f_pred = f_pred[:, :target.shape[-1]]
        p_pred = p_pred[:, :target.shape[-1]]
        o_pred = o_pred[:, :target.shape[-1]]
        if ds == 0:
            all_tar = target
            all_pred = f_pred
            pitch_tar = target_p
            pp_pred = p_pred
            onset_tar = target_o
            oo_pred = o_pred
            ds += 1
        else:
            all_tar = np.concatenate([all_tar, target], axis=-1)
            all_pred = np.concatenate([all_pred, f_pred], axis=-1)
            pitch_tar = np.concatenate([pitch_tar, target_p], axis=-1)
            pp_pred = np.concatenate([pp_pred, p_pred], axis=-1)
            onset_tar = np.concatenate([onset_tar, target_o], axis=-1)
            oo_pred = np.concatenate([oo_pred, o_pred], axis=-1)
    threshold = 0.5
    pred_IPT = all_pred
    pred_IPT[pred_IPT > threshold] = 1
    pred_IPT[pred_IPT <= threshold] = 0
    pred_pitch = pp_pred
    pred_pitch[pred_pitch > threshold] = 1
    pred_pitch[pred_pitch <= threshold] = 0
    pred_onset = oo_pred
    pred_onset[pred_onset > threshold] = 1
    pred_onset[pred_onset <= threshold] = 0
    target_IPT = all_tar
    tar_pitch = pitch_tar
    tar_onset = onset_tar

    # compute metrics
    metrics_no_infer, _ = compute_metrics_with_note_no_infer(pred_IPT, target_IPT,pred_onset, tar_onset)
    metrics, roll = compute_metrics_with_note(pred_IPT, target_IPT, pred_onset, tar_onset)
    print("The result before post-processing：")
    print("IPT_frame_precision:", metrics_no_infer['metric/IPT_frame/precision'])
    print("IPT_frame_recall:", metrics_no_infer['metric/IPT_frame/recall'])
    print("IPT_frame_f1:", metrics_no_infer['metric/IPT_frame/f1'])
    print("IPT_frame_accuracy:", metrics_no_infer['metric/IPT_frame/accuracy'])
    print("IPT_note_precision:", metrics_no_infer['metric/note/precision'])
    print("IPT_note_recall:", metrics_no_infer['metric/note/recall'])
    print("IPT_note_f1:", metrics_no_infer['metric/note/f1'])
    print("The result after post-processing：")
    print("IPT_frame_precision:", metrics['metric/IPT_frame/precision'])
    print("IPT_frame_recall:", metrics['metric/IPT_frame/recall'])
    print("IPT_frame_f1:", metrics['metric/IPT_frame/f1'])
    print("IPT_frame_accuracy:", metrics['metric/IPT_frame/accuracy'])
    print("IPT_note_precision:", metrics['metric/note/precision'])
    print("IPT_note_recall:", metrics['metric/note/recall'])
    print("IPT_note_f1:", metrics['metric/note/f1'])

    metrics0, _ = compute_metrics_with_note(pred_IPT[0, :].reshape((1, -1)), target_IPT[0, :].reshape((1, -1)), pred_onset, tar_onset)
    metrics1, _ = compute_metrics_with_note(pred_IPT[1, :].reshape((1, -1)), target_IPT[1, :].reshape((1, -1)), pred_onset, tar_onset)
    metrics2, _ = compute_metrics_with_note(pred_IPT[2, :].reshape((1, -1)), target_IPT[2, :].reshape((1, -1)), pred_onset, tar_onset)
    metrics3, _ = compute_metrics_with_note(pred_IPT[3, :].reshape((1, -1)), target_IPT[3, :].reshape((1, -1)), pred_onset, tar_onset)
    metrics4, _ = compute_metrics_with_note(pred_IPT[4, :].reshape((1, -1)), target_IPT[4, :].reshape((1, -1)), pred_onset, tar_onset)
    metrics5, _ = compute_metrics_with_note(pred_IPT[5, :].reshape((1, -1)), target_IPT[5, :].reshape((1, -1)), pred_onset, tar_onset)
    metrics6, _ = compute_metrics_with_note(pred_IPT[6, :].reshape((1, -1)), target_IPT[6, :].reshape((1, -1)), pred_onset, tar_onset)

    print("vibrato_frame_f1:", metrics0['metric/IPT_frame/f1'])
    print("vibrato_note_f1:", metrics0['metric/note/f1'])
    print("plucks_frame_f1:", metrics1['metric/IPT_frame/f1'])
    print("plucks_note_f1:", metrics1['metric/note/f1'])
    print("UP_frame_f1:", metrics2['metric/IPT_frame/f1'])
    print("UP_note_f1:", metrics2['metric/note/f1'])
    print("DP_frame_f1:", metrics3['metric/IPT_frame/f1'])
    print("DP_note_f1:", metrics3['metric/note/f1'])
    print("glissando_frame_f1:", metrics4['metric/IPT_frame/f1'])
    print("glissando_note_f1:", metrics4['metric/note/f1'])
    print("tremolo_frame_f1:", metrics5['metric/IPT_frame/f1'])
    print("tremolo_note_f1:", metrics5['metric/note/f1'])
    print("PN_frame_f1:", metrics6['metric/IPT_frame/f1'])
    print("PN_note_f1:", metrics6['metric/note/f1'])
    macro_frame_f1 = float(metrics0['metric/IPT_frame/f1'][0] + metrics1['metric/IPT_frame/f1'][0] +
                      metrics2['metric/IPT_frame/f1'][0] + metrics3['metric/IPT_frame/f1'][0] +
                      metrics4['metric/IPT_frame/f1'][0] + metrics5['metric/IPT_frame/f1'][0] +
                      metrics6['metric/IPT_frame/f1'][0])/7.0
    macro_note_f1 = float(metrics0['metric/note/f1'][0] + metrics1['metric/note/f1'][0] +
                     metrics2['metric/note/f1'][0] +metrics3['metric/note/f1'][0] +
                     metrics4['metric/note/f1'][0] +metrics5['metric/note/f1'][0] +
                     metrics6['metric/note/f1'][0])/7.0
    print("macro_frame_f1:",macro_frame_f1)
    print("macro_note_f1:",macro_note_f1)

start_test()
