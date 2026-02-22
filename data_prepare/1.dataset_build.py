import os
import pdb
import numpy as np
from tqdm import tqdm
from itertools import combinations, combinations_with_replacement
import random

rng = random.Random(42)


def list_edit_distance(list1, list2):
    len1, len2 = len(list1), len(list2)
    
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j],
                                dp[i][j - 1],
                                dp[i - 1][j - 1]) + 1
    
    return dp[len1][len2]


def readList(data_list):

    phn2path = {}

    with open(data_list) as f:
        lines = f.readlines()
    
    for line in tqdm(lines):
        sample = np.load(line.strip(), allow_pickle=True).item()
        phn_anc = str("-".join(sample['anc_phn_list']))
        phn_com = str("-".join(sample['com_phn_list']))

        if phn_anc not in phn2path:
            phn2path[phn_anc] = set()
        phn2path[phn_anc].add((sample['anc_wav_path'], sample['anc_text'], sample['anc_text_fea'], sample['anc_vide_fea_path'], sample['anc_audi_fea_path'], sample['anc_lip_path'], sample['anc_wav_path']))

        if phn_com not in phn2path:
            phn2path[phn_com] = set()
        phn2path[phn_com].add((sample['com_wav_path'], sample['com_text'], sample['com_text_fea'], sample['com_vide_fea_path'], sample['com_audi_fea_path'], sample['com_lip_path'], sample['com_wav_path']))

    phn2path = {key: list(value_set) for key, value_set in phn2path.items()}

    return phn2path

def buildNeg(phn2path, npy_save, total_easy_negative = 200000, total_hard_negative = 200000):

    positive_count = 0
    easy_negative_count = 0
    hard_negative_count = 0

    phn_set = list(phn2path.keys())
    
    num = 21 if '/train/' in npy_save else 11
    
    for phn1 in tqdm(phn_set):

        phn2_all = rng.sample(phn_set, k = num)

        if phn1 in phn2_all:
            phn2_all.remove(phn1)
        else:
            phn2_all.pop(-1)

        for phn2 in phn2_all:
            phn1_list = phn1.split('-')
            phn2_list = phn2.split('-')
            distance = list_edit_distance(phn1_list, phn2_list)

            if distance == 0:
                label = 1
                datatype = "positive"
                positive_count += 1
            elif distance in [1,2,3] and distance < max(len(phn1_list), len(phn2_list)) / 2:
                label = 0
                datatype = "hard_negative"
                hard_negative_count += 1
            else:
                label = 0
                datatype = "easy_negative"
                easy_negative_count += 1

            anc_indices = rng.sample(range(len(phn2path[phn1])), min(len(phn2path[phn1]), 2))
            com_indices = rng.sample(range(len(phn2path[phn2])), min(len(phn2path[phn2]), 2))

            for i in anc_indices:
                for j in com_indices:
                    data_dict={
                            'anc_phn_list':phn1_list,
                            'anc_text':phn2path[phn1][i][1], 
                            'anc_text_fea':phn2path[phn1][i][2],
                            'anc_vide_fea_path':phn2path[phn1][i][3], 
                            'anc_audi_fea_path':phn2path[phn1][i][4], 
                            'anc_lip_path':phn2path[phn1][i][5], 
                            'anc_wav_path':phn2path[phn1][i][6], 
                            'com_phn_list':phn2_list,
                            'com_text':phn2path[phn2][j][1], 
                            'com_text_fea':phn2path[phn2][j][2],
                            'com_vide_fea_path':phn2path[phn2][j][3], 
                            'com_audi_fea_path':phn2path[phn2][j][4], 
                            'com_lip_path':phn2path[phn2][j][5],
                            'com_wav_path':phn2path[phn2][j][6],
                            'type':datatype, 
                            'label':label, 
                            }
                name = '_'.join(phn2path[phn1][i][6].split('/')[-2:]).split('.')[0] + '+' + '_'.join(phn2path[phn2][j][6].split('/')[-2:]).split('.')[0]
                save_path = npy_save + name + '.npy'

                directory = os.path.dirname(save_path)
                os.makedirs(directory, exist_ok=True)
                if datatype == "hard_negative" and hard_negative_count <= total_hard_negative:
                    np.save(save_path, data_dict)
                elif datatype == "easy_negative" and easy_negative_count <= total_easy_negative:
                    np.save(save_path, data_dict)
                else:
                    print("Wrong datatype")


def buildPos(phn2path, npy_save, total_positive = 100000):

    positive_count = 0
    phn_set = list(phn2path.keys())
    num = 10
    
    for phn in tqdm(phn_set):
        phn_list = phn.split('-')
        anc_indices = rng.sample(range(len(phn2path[phn])), k = min(len(phn2path[phn]), num))
        com_indices = rng.sample(range(len(phn2path[phn])), k = min(len(phn2path[phn]), num))
        label = 1
        seen = []
        for i in anc_indices:
            for j in com_indices:
                group = (i, j)
                if group in seen:
                    continue
                seen.append(group)
                data_dict={
                        'anc_phn_list':phn_list,
                        'anc_text':phn2path[phn][i][1], 
                        'anc_text_fea':phn2path[phn][i][2],
                        'anc_vide_fea_path':phn2path[phn][i][3], 
                        'anc_audi_fea_path':phn2path[phn][i][4], 
                        'anc_lip_path':phn2path[phn][i][5], 
                        'anc_wav_path':phn2path[phn][i][6], 
                        'com_phn_list':phn_list,
                        'com_text':phn2path[phn][j][1], 
                        'com_text_fea':phn2path[phn][j][2],
                        'com_vide_fea_path':phn2path[phn][j][3], 
                        'com_audi_fea_path':phn2path[phn][j][4], 
                        'com_lip_path':phn2path[phn][j][5],
                        'com_wav_path':phn2path[phn][j][6],
                        'type':"positive", 
                        'label':label, 
                        }
                name = '_'.join(phn2path[phn][i][6].split('/')[-2:]).split('.')[0] + '+' + '_'.join(phn2path[phn][j][6].split('/')[-2:]).split('.')[0]
                save_path = npy_save + name + '.npy'
                positive_count += 1
                directory = os.path.dirname(save_path)
                os.makedirs(directory, exist_ok=True)
                if positive_count <= total_positive:
                    np.save(save_path, data_dict)
                else:
                    continue


train_list = "/path/train_offline_tva_1word.scp"
test_list = "/path/test_offline_all_tva_1word.scp"

npy_save_pos_train = f'/path/npy/pos_train/'
npy_save_neg_train = f'/path/npy/neg_train/'

npy_save_pos_test_inset = f'/path/npy/pos_test_inset/'
npy_save_pos_test_outset = f'/path/npy/pos_test_outset/'

npy_save_neg_test_inset = f'/path/npy/neg_test_inset/'
npy_save_neg_test_outset = f'/path/npy/neg_test_outset/'

train_phn2path = readList(train_list)
test_phn2path = readList(test_list)

test_phn2path_inset = {}
test_phn2path_outset = {}

for key, value in test_phn2path.items():
    if key in train_phn2path:
        test_phn2path_inset[key] = value
    else:
        test_phn2path_outset[key] = value
del test_phn2path

buildNeg(train_phn2path, npy_save_neg_train)
buildPos(train_phn2path, npy_save_pos_train)
del train_phn2path

buildNeg(test_phn2path_inset, npy_save_neg_test_inset)
buildPos(test_phn2path_inset, npy_save_pos_test_inset)
del test_phn2path_inset

buildNeg(test_phn2path_outset, npy_save_neg_test_outset)
buildPos(test_phn2path_outset, npy_save_pos_test_outset)
del test_phn2path_outset
