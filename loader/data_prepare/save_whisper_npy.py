from pathlib import Path
import sys, os
import pandas as pd
from torch.utils.data import Dataset
import torch

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import whisper
import difflib
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from dataload import LibriPhraseDataset

a = LibriPhraseDataset()
# a = LibriPhraseDataset(types = 'easy')
# a = LibriPhraseDataset(types = 'hard')
data = a.get_data()

anchor_wav = data['anchor_wav'].values
anchor_text = data['anchor_text'].values
comparison_text = data['comparison_text'].values
comparison_wav = data['comparison_wav'].values
label = data['label'].values
datatype = data['type'].values

y_scores = []
text_tmp = []
new_file_scp = []
# device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = whisper.load_model("/train20/intern/permanent/kwli2/udkws/tmp1/whisper_models/tiny.pt")
model = model.to(device)

# save_dir = '/train20/intern/temporary/kwli2/udkws/interfea/whisperenc/100h/'
save_dir = '/train20/intern/temporary/kwli2/udkws/interfea/whisperenc/360h/4word/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for i in tqdm(range(len(anchor_wav))):
    audio_path_an = anchor_wav[i]
    audio = whisper.load_audio(audio_path_an)
    # noise
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    mel = mel.unsqueeze(0)
    anchor_feature = model.encoder(mel)  # [1, 1500, 384] #[t,c]
    anchor_feature = anchor_feature.cpu().detach()[:, :100, :]

    audio_path_com = comparison_wav[i]
    audio_com = whisper.load_audio(audio_path_com)
    # noise
    audio_com = whisper.pad_or_trim(audio_com)
    mel_com = whisper.log_mel_spectrogram(audio_com).to(model.device)
    mel_com = mel_com.unsqueeze(0)
    com_feature = model.encoder(mel_com)  # [1, 1500, 384] #[t,c]
    com_feature = com_feature.cpu().detach()[:, :100, :]

    data_dict={'anchor_speech':anchor_feature, 'comparison_speech':com_feature, 'anchor_text':anchor_text[i], 'comparison_text':comparison_text[i], 'type':datatype[i], 'label':label[i], 'anchor_path':anchor_wav[i], 'comparison_path':comparison_wav[i]}
    # data_dict={'clean_speech':a_feature, 'noisy_speech':noisy_speech, 'type':datatype[i], 'label':label[i], 'speech_labels':wav_label[i], 'text_labels':text[i]}
    name = anchor_wav[i].split('/')[-1].split('.')[0] + '_' + comparison_wav[i].split('/')[-1].split('.')[0]
    save_path = save_dir + name + '.npy'
    np.save(save_path, data_dict)

    new_file_scp.append(save_path) 

with open('/train20/intern/permanent/kwli2/udkws/tmp1/npyfile/testeasy_all_360h_4word.scp', 'w') as output:
    output.writelines(it + '\n' for it in new_file_scp)
