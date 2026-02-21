import math, os, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
# import Levenshtein
# from multiprocessing import Pool
from scipy.io import wavfile

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import librosa
from tqdm import tqdm
import math
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname('/train20/intern/permanent/kwli2/udkws/attakwss'))
from g2p.g2p_en.g2p import G2p

g2p = G2p()
scp_path_at = '/train20/intern/permanent/kwli2/udkws/tmp1/npyfileTest'
files_scp = []
train_csv = ['all_100h', ] #testeasy_all_500h all_100h testhard_all_500h all_360h
for db in train_csv:
    csv_list = [str(x) for x in Path(scp_path_at).rglob('*' + db + '*word*')]
    files_scp_tmp = []
    for n_word in csv_list:
        print(">> processing : {} ".format(n_word))
        files_scp_tmp = []

        with open(n_word) as f:
            lines = f.readlines()
        files_scp_tmp = [line.strip() for line in lines]
        # files_scp_tmp = files_scp_tmp[:200]
        files_scp = files_scp + files_scp_tmp
# files_scp
len1 = len(files_scp)
for i in tqdm(range(len(files_scp))):

    data_path = files_scp[i]
    feature_data = np.load(data_path, allow_pickle=True).item()
    # comg2p = g2p(comparison_text)
    anchor_text = feature_data['anchor_text']
    comparison_text = feature_data['comparison_text']
    # comg2p = g2p(comparison_text)
    # com_embed = g2p.embedding(comparison_text)
    # com_embed = torch.from_numpy(com_embed)
    # feature_data['com_g2p'] = comg2p
    # feature_data['com_g2p_emb'] = com_embed

    ang2p = g2p(anchor_text)
    an_embed = g2p.embedding(anchor_text)
    an_embed = torch.from_numpy(an_embed)
    feature_data['an_g2p'] = ang2p
    feature_data['an_g2p_emb'] = an_embed
    # path = 'sfsfd.npy'
    # save_path = '/train20/intern/temporary/kwli2/udkws/interfea/whis2/' + name

    np.save(data_path, feature_data)

    # feature_data['anchor_speech'] = feature_data['anchor_speech'].cpu().detach()
    # feature_data['anchor_speech'] = feature_data['anchor_speech'][:, :100, :]
    # feature_data['comparison_speech'] = feature_data['comparison_speech'].cpu().detach()
    # feature_data['comparison_speech'] = feature_data['comparison_speech'][:, :100, :]

    # name = data_path.split('/')[-1] 
    # save_path = '/train20/intern/temporary/kwli2/udkws/interfea/whis2/' + name
    # np.save(data_path, feature_data)

#CUDA_VISIBLE_DEVICES=1 nohup python g2p_npy.py > t500e_1.log 2>&1 &
