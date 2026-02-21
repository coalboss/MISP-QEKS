from praatio import textgrid
import torch
import math
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import random

phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                                'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                                'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                                'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
                                ' ']
p2idx = {p: idx for idx, p in enumerate(phonemes)}
idx2p = {idx: p for idx, p in enumerate(phonemes)}

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


def find_common_indices(list1, list2):
    common_indices_1 = []
    common_indices_2 = []
    
    start_idx2 = 0

    for idx1, elem1 in enumerate(list1):
        for idx2 in range(start_idx2, len(list2)):
            if elem1 == list2[idx2]:
                common_indices_1.append(idx1)
                common_indices_2.append(idx2)
                start_idx2 = idx2 + 1
                break

    return common_indices_1, common_indices_2


def maxpool_with_gaussian(av_embedding, text_embedding):
    T_, C_ = av_embedding.shape
    kernel = torch.from_numpy(np.array([0.25, 0.5, 0.25])).float().reshape(-1)
    kernel = kernel.cuda()
    radius = len(kernel) // 2
    max_dis = -1
    for i in range(av_embedding.shape[0]):
        cur_distance = F.mse_loss(av_embedding[i], text_embedding)
        if cur_distance > max_dis:
            max_dis = cur_distance
            center_indice = i
    start_embd = max(0, center_indice - radius)
    end_embd = min(T_ - 1, center_indice + radius)
    embd_part = av_embedding[start_embd:end_embd + 1]  # shape: [valid_T, C]

    len_valid = end_embd - start_embd + 1
    start_a = radius - (center_indice - start_embd)
    end_a = start_a + len_valid
    kernel_part = kernel[start_a:end_a]  # shape: [valid_T]

    max_pooled_av_fea = (kernel_part[:, None] * embd_part).sum(axis=0)  # shape [C]

    return max_pooled_av_fea






def aa_exatphone_pool(an_audio_fea, an_audio_mask, anc_textgrid_path, com_audio_fea, com_audio_mask, com_textgrid_path, label):
    phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
                                    ' ']
    p2idx = {p: idx for idx, p in enumerate(phonemes)}
    B, T_an, _ = an_audio_fea.shape
    len_audioan = an_audio_mask.sum(dim = 1)

    B, T_com, _ = com_audio_fea.shape
    len_audiocom = com_audio_mask.sum(dim = 1)
    compare_label = label
    bath_list = []
    bath_text = []
    bath_emb = []
    mask = []

    com_bath_list = []
    com_bath_text = []
    com_bath_emb = []
    com_mask = []
    for i in range(B):
        if compare_label[i] == 0:
            continue
        tg = textgrid.openTextgrid(anc_textgrid_path[i], includeEmptyIntervals=True)
        maxtime = tg.maxTimestamp
        tier_name = "phones"
        word_list = []
        an_lab_p = []
        len_a = int(len_audioan[i])
        if tier_name in tg.tierNames:
            tier = tg._tierDict[tier_name]
            for interval in tier.entries:
                bgn_time, end_time, label = interval
                
                if label == '' or label == 'sp' or label == 'spn' or label == 'sil' or label == 'noise':
                    continue
                else:
                    an_lab_p.extend([p2idx[label]])

                    # FA结果和音视频帧分辨率不同，起止点不会完全对齐，这里取交集保证所取音视频特征中不包含其它音素；如果长度仅有一帧则进行额外处理
                    bgn_embd_float = max(0, len_a * (bgn_time / maxtime))
                    end_embd_float = max(len_a * (end_time / maxtime), 1)

                    bgn_embd = math.floor(bgn_embd_float)
                    end_embd = math.ceil(end_embd_float)

                    if bgn_embd == end_embd:
                        if 1 - (bgn_embd_float % 1) >= end_embd_float % 1:
                            bgn_embd -= 1
                        else:
                            end_embd += 1
                    # pool phone
                    phone_fea = an_audio_fea[i, bgn_embd:end_embd, :]
                    phone_fea = phone_fea.transpose(0, 1)
                    pooled_phone = F.adaptive_max_pool1d(phone_fea, 1).transpose(0, 1)
                    word_list.append(pooled_phone)

            wordphn_embd_an = pad_sequence(word_list, batch_first=True, padding_value=0.0).squeeze(1)

        tg = textgrid.openTextgrid(com_textgrid_path[i], includeEmptyIntervals=True)
        maxtime = tg.maxTimestamp
        tier_name = "phones"
        word_list = []
        com_lab_p = []
        len_a = int(len_audiocom[i])
        if tier_name in tg.tierNames:
            tier = tg._tierDict[tier_name]
            for interval in tier.entries:
                bgn_time, end_time, label = interval
                
                if label == '' or label == 'sp' or label == 'spn' or label == 'sil' or label == 'noise':
                    continue
                else:
                    com_lab_p.extend([p2idx[label]])

                    # FA结果和音视频帧分辨率不同，起止点不会完全对齐，这里取交集保证所取音视频特征中不包含其它音素；如果长度仅有一帧则进行额外处理
                    bgn_embd_float = max(0, len_a * (bgn_time / maxtime))
                    end_embd_float = max(len_a * (end_time / maxtime), 1)

                    bgn_embd = math.floor(bgn_embd_float)
                    end_embd = math.ceil(end_embd_float)

                    if bgn_embd == end_embd:
                        if 1 - (bgn_embd_float % 1) >= end_embd_float % 1:
                            bgn_embd -= 1
                        else:
                            end_embd += 1
                    # pool phone
                    phone_fea = com_audio_fea[i, bgn_embd:end_embd, :]
                    phone_fea = phone_fea.transpose(0, 1)
                    pooled_phone = F.adaptive_avg_pool1d(phone_fea, 1).transpose(0, 1)
                    word_list.append(pooled_phone)

            wordphn_embd_com = pad_sequence(word_list, batch_first=True, padding_value=0.0).squeeze(1)

            if com_lab_p != an_lab_p:
                continue
            bath_list.append(wordphn_embd_an)
            bath_text.extend(an_lab_p)
            com_bath_list.append(wordphn_embd_com)

    if len(bath_list) != 0:
        bath_text = torch.tensor(bath_text)
        an_audio_phnfea = torch.cat(bath_list, dim=0)
        com_audio_phnfea = torch.cat(com_bath_list, dim=0)

        return an_audio_phnfea, bath_text, com_audio_phnfea, len(bath_list)
    else:
        return 1

def vv_exatphone_pool(anc_video_fea, anc_lip_mask, anc_textgrid_path, com_video_fea, com_lip_mask, com_textgrid_path, label):
    phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
                                    ' ']
    p2idx = {p: idx for idx, p in enumerate(phonemes)}
    B, T_an, _ = anc_video_fea.shape
    len_video_anc = anc_lip_mask.sum(dim = 1)

    B, T_com, _ = com_video_fea.shape
    len_video_com = com_lip_mask.sum(dim = 1)
    compare_label = label
    bath_list = []
    bath_text = []

    com_bath_list = []

    for i in range(B):
        if compare_label[i] == 0:
            continue
        tg = textgrid.openTextgrid(anc_textgrid_path[i], includeEmptyIntervals=True)
        maxtime = tg.maxTimestamp
        tier_name = "phones"
        word_list = []
        an_lab_p = []
        len_a = int(len_video_anc[i])
        if tier_name in tg.tierNames:
            tier = tg._tierDict[tier_name]
            for interval in tier.entries:
                bgn_time, end_time, label = interval
                if label == '' or label == 'sp' or label == 'spn' or label == 'sil' or label == 'noise':
                    continue
                else:
                    an_lab_p.extend([p2idx[label]])

                    # FA结果和音视频帧分辨率不同，起止点不会完全对齐，这里取交集保证所取音视频特征中不包含其它音素；如果长度仅有一帧则进行额外处理
                    bgn_embd_float = max(0, len_a * (bgn_time / maxtime))
                    end_embd_float = max(len_a * (end_time / maxtime), 1)

                    bgn_embd = math.floor(bgn_embd_float)
                    end_embd = math.ceil(end_embd_float)

                    if bgn_embd == end_embd:
                        if 1 - (bgn_embd_float % 1) >= end_embd_float % 1:
                            bgn_embd -= 1
                        else:
                            end_embd += 1                 

                    # pool phone
                    phone_fea = anc_video_fea[i, bgn_embd:end_embd, :]
                    phone_fea = phone_fea.transpose(0, 1)
                    pooled_phone = F.adaptive_max_pool1d(phone_fea, 1).transpose(0, 1)
                    word_list.append(pooled_phone)

            wordphn_embd_an = pad_sequence(word_list, batch_first=True, padding_value=0.0).squeeze(1)

        tg = textgrid.openTextgrid(com_textgrid_path[i], includeEmptyIntervals=True)
        maxtime = tg.maxTimestamp
        tier_name = "phones"
        word_list = []
        com_lab_p = []
        len_a = int(len_video_com[i])
        if tier_name in tg.tierNames:
            tier = tg._tierDict[tier_name]
            for interval in tier.entries:
                bgn_time, end_time, label = interval
                
                if label == '' or label == 'sp' or label == 'spn' or label == 'sil' or label == 'noise':
                    continue
                else:
                    com_lab_p.extend([p2idx[label]])

                    # FA结果和音视频帧分辨率不同，起止点不会完全对齐，这里取交集保证所取音视频特征中不包含其它音素；如果长度仅有一帧则进行额外处理
                    bgn_embd_float = max(0, len_a * (bgn_time / maxtime))
                    end_embd_float = max(len_a * (end_time / maxtime), 1)

                    bgn_embd = math.floor(bgn_embd_float)
                    end_embd = math.ceil(end_embd_float)

                    if bgn_embd == end_embd:
                        if 1 - (bgn_embd_float % 1) >= end_embd_float % 1:
                            bgn_embd -= 1
                        else:
                            end_embd += 1
                    # pool phone
                    phone_fea = com_video_fea[i, bgn_embd:end_embd, :]
                    phone_fea = phone_fea.transpose(0, 1)
                    pooled_phone = F.adaptive_avg_pool1d(phone_fea, 1).transpose(0, 1)
                    word_list.append(pooled_phone)

            wordphn_embd_com = pad_sequence(word_list, batch_first=True, padding_value=0.0).squeeze(1)

            if com_lab_p != an_lab_p:
                continue
            bath_list.append(wordphn_embd_an)
            bath_text.extend(an_lab_p)
            com_bath_list.append(wordphn_embd_com)

    if len(bath_list) != 0:
        bath_text = torch.tensor(bath_text)
        anc_video_phnfea = torch.cat(bath_list, dim=0)
        com_video_phnfea = torch.cat(com_bath_list, dim=0)

        return anc_video_phnfea, bath_text, com_video_phnfea, len(bath_list)
    else:
        return 1

def ta_exatphone_pool(audio_fea, audio_mask, textgrid_path, comg2p, pad_anphn, mask_textan):
    phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
                                    ' ']
    p2idx = {p: idx for idx, p in enumerate(phonemes)}
    B, T, _ = audio_fea.shape
    len_audio = audio_mask.sum(dim = 1)
    
    bath_list = []
    bath_text = []
    bath_emb = []
    mask = []
    for i in range(B):
        tg = textgrid.openTextgrid(textgrid_path[i], includeEmptyIntervals=True)
        maxtime = tg.maxTimestamp

        tier_name = "phones"
        word_list = []
        lab_p = []
        len_a = int(len_audio[i])
        if tier_name in tg.tierNames:
            tier = tg._tierDict[tier_name]
            for interval in tier.entries:
                bgn_time, end_time, label = interval      
                if label == '' or label == 'sp' or label == 'spn' or label == 'sil' or label == 'noise':
                    continue
                else:
                    lab_p.append(p2idx[label])
                    # FA结果和音视频帧分辨率不同，起止点不会完全对齐，这里取交集保证所取音视频特征中不包含其它音素；如果长度仅有一帧则进行额外处理
                    bgn_embd_float = max(0, len_a * (bgn_time / maxtime))
                    end_embd_float = max(len_a * (end_time / maxtime), 1)

                    bgn_embd = math.floor(bgn_embd_float)
                    end_embd = math.ceil(end_embd_float)

                    if bgn_embd == end_embd:
                        if 1 - (bgn_embd_float % 1) >= end_embd_float % 1:
                            bgn_embd -= 1
                        else:
                            end_embd += 1

                    phone_fea = audio_fea[i, bgn_embd:end_embd, :]
                    phone_fea = phone_fea.transpose(0, 1)
                    pooled_phone = F.adaptive_avg_pool1d(phone_fea, 1).transpose(0, 1)
                    word_list.append(pooled_phone)

            wordphn_embd = pad_sequence(word_list, batch_first=True, padding_value=0.0).squeeze(1)

            masktab = comg2p[i] != 71

            label_text = comg2p[i][comg2p[i] != 0]
            mask_t = torch.bitwise_and(masktab, mask_textan[i].bool().squeeze())
            label_text = label_text[label_text!=71]
            
            if list(label_text) == lab_p:
                bath_list.append(wordphn_embd)
                bath_text.append(label_text)
                bath_emb.append(pad_anphn[i])
                mask.append(mask_t) #embd mask

    if len(bath_list) != 0:
        audio_phnfea = torch.cat(bath_list, dim=0)
        phn_text = torch.cat(bath_text, dim=0)
        phn_mask = torch.cat(mask, dim=0)
        phn_text_embd = torch.cat(bath_emb, dim=0)

        # 去除mask为0部分的text embedding，shape [T, 128]
        phn_text_embd2 = torch.masked_select(phn_text_embd, phn_mask.bool().unsqueeze(-1)).unsqueeze(1).view(-1,128)

        return audio_phnfea, phn_text, phn_text_embd2, len(bath_list)
    else:
        return 1

def tv_exatphone_pool(video_fea, video_mask, textgrid_path, comg2p, pad_anphn, mask_textan):
    phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
                                    ' ']
    p2idx = {p: idx for idx, p in enumerate(phonemes)}
    B, T, _ = video_fea.shape
    len_video = video_mask.sum(dim = 1)
    
    bath_list = []
    bath_text = []
    bath_emb = []
    mask = []
    for i in range(B):
        tg = textgrid.openTextgrid(textgrid_path[i], includeEmptyIntervals=True)
        maxtime = tg.maxTimestamp

        tier_name = "phones"
        word_list = []
        lab_p = []
        len_a = int(len_video[i])
        if tier_name in tg.tierNames:
            tier = tg._tierDict[tier_name]
            for interval in tier.entries:
                bgn_time, end_time, label = interval      
                if label == '' or label == 'sp' or label == 'spn' or label == 'sil' or label == 'noise':
                    continue
                else:
                    lab_p.append(p2idx[label])

                    # FA结果和音视频帧分辨率不同，起止点不会完全对齐，这里取交集保证所取音视频特征中不包含其它音素；如果长度仅有一帧则进行额外处理
                    bgn_embd_float = max(0, len_a * (bgn_time / maxtime))
                    end_embd_float = max(len_a * (end_time / maxtime), 1)

                    bgn_embd = math.floor(bgn_embd_float)
                    end_embd = math.ceil(end_embd_float)

                    if bgn_embd == end_embd:
                        if 1 - (bgn_embd_float % 1) >= end_embd_float % 1:
                            bgn_embd -= 1
                        else:
                            end_embd += 1

                    phone_fea = video_fea[i, bgn_embd:end_embd, :]
                    phone_fea = phone_fea.transpose(0, 1)
                    pooled_phone = F.adaptive_avg_pool1d(phone_fea, 1).transpose(0, 1)
                    word_list.append(pooled_phone)

            wordphn_embd = pad_sequence(word_list, batch_first=True, padding_value=0.0).squeeze(1)

            masktab = comg2p[i] != 71

            label_text = comg2p[i][comg2p[i] != 0]
            mask_t = torch.bitwise_and(masktab, mask_textan[i].bool().squeeze())
            label_text = label_text[label_text!=71]
            
            if list(label_text) == lab_p:
                bath_list.append(wordphn_embd)
                bath_text.append(label_text)
                bath_emb.append(pad_anphn[i])
                mask.append(mask_t) #embd mask

    if len(bath_list) != 0:
        video_phnfea = torch.cat(bath_list, dim=0)
        phn_text = torch.cat(bath_text, dim=0)
        phn_mask = torch.cat(mask, dim=0)
        phn_text_embd = torch.cat(bath_emb, dim=0)

        # 去除mask为0部分的text embedding，shape [T, 128]
        phn_text_embd2 = torch.masked_select(phn_text_embd, phn_mask.bool().unsqueeze(-1)).unsqueeze(1).view(-1,128)

        return video_phnfea, phn_text, phn_text_embd2, len(bath_list)
        # (T, 384) (T) (T,256)
    else:
        return 1

def va_exatphone_pool(anc_video_fea, anc_lip_mask, anc_textgrid_path, com_audio_fea, com_audi_mask, com_textgrid_path, label):
    phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
                                    ' ']
    p2idx = {p: idx for idx, p in enumerate(phonemes)}
    B, T_an, _ = anc_video_fea.shape
    len_video_anc = anc_lip_mask.sum(dim = 1)

    B, T_com, _ = com_audio_fea.shape
    len_video_com = com_audi_mask.sum(dim = 1)
    compare_label = label
    bath_list = []
    bath_text = []

    com_bath_list = []

    for i in range(B):
        if compare_label[i] == 0:
            continue
        tg = textgrid.openTextgrid(anc_textgrid_path[i], includeEmptyIntervals=True)
        maxtime = tg.maxTimestamp
        tier_name = "phones"
        word_list = []
        an_lab_p = []
        len_a = int(len_video_anc[i])
        if tier_name in tg.tierNames:
            tier = tg._tierDict[tier_name]
            for interval in tier.entries:
                bgn_time, end_time, label = interval
                if label == '' or label == 'sp' or label == 'spn' or label == 'sil' or label == 'noise':
                    continue
                else:
                    an_lab_p.extend([p2idx[label]])

                    # FA结果和音视频帧分辨率不同，起止点不会完全对齐，这里取交集保证所取音视频特征中不包含其它音素；如果长度仅有一帧则进行额外处理
                    bgn_embd_float = max(0, len_a * (bgn_time / maxtime))
                    end_embd_float = max(len_a * (end_time / maxtime), 1)

                    bgn_embd = math.floor(bgn_embd_float)
                    end_embd = math.ceil(end_embd_float)

                    if bgn_embd == end_embd:
                        if 1 - (bgn_embd_float % 1) >= end_embd_float % 1:
                            bgn_embd -= 1
                        else:
                            end_embd += 1                 

                    # pool phone
                    phone_fea = anc_video_fea[i, bgn_embd:end_embd, :]
                    phone_fea = phone_fea.transpose(0, 1)
                    pooled_phone = F.adaptive_max_pool1d(phone_fea, 1).transpose(0, 1)
                    word_list.append(pooled_phone)

            wordphn_embd_an = pad_sequence(word_list, batch_first=True, padding_value=0.0).squeeze(1)

        tg = textgrid.openTextgrid(com_textgrid_path[i], includeEmptyIntervals=True)
        maxtime = tg.maxTimestamp
        tier_name = "phones"
        word_list = []
        com_lab_p = []
        len_a = int(len_video_com[i])
        if tier_name in tg.tierNames:
            tier = tg._tierDict[tier_name]
            for interval in tier.entries:
                bgn_time, end_time, label = interval
                
                if label == '' or label == 'sp' or label == 'spn' or label == 'sil' or label == 'noise':
                    continue
                else:
                    com_lab_p.extend([p2idx[label]])

                    # FA结果和音视频帧分辨率不同，起止点不会完全对齐，这里取交集保证所取音视频特征中不包含其它音素；如果长度仅有一帧则进行额外处理
                    bgn_embd_float = max(0, len_a * (bgn_time / maxtime))
                    end_embd_float = max(len_a * (end_time / maxtime), 1)

                    bgn_embd = math.floor(bgn_embd_float)
                    end_embd = math.ceil(end_embd_float)

                    if bgn_embd == end_embd:
                        if 1 - (bgn_embd_float % 1) >= end_embd_float % 1:
                            bgn_embd -= 1
                        else:
                            end_embd += 1
                    # pool phone
                    phone_fea = com_audio_fea[i, bgn_embd:end_embd, :]
                    phone_fea = phone_fea.transpose(0, 1)
                    pooled_phone = F.adaptive_avg_pool1d(phone_fea, 1).transpose(0, 1)
                    word_list.append(pooled_phone)

            wordphn_embd_com = pad_sequence(word_list, batch_first=True, padding_value=0.0).squeeze(1)

            if com_lab_p != an_lab_p:
                continue
            bath_list.append(wordphn_embd_an)
            bath_text.extend(an_lab_p)
            com_bath_list.append(wordphn_embd_com)

    if len(bath_list) != 0:
        bath_text = torch.tensor(bath_text)
        anc_video_phnfea = torch.cat(bath_list, dim=0)
        com_audio_phnfea = torch.cat(com_bath_list, dim=0)

        return anc_video_phnfea, bath_text, com_audio_phnfea, len(bath_list)
    else:
        return 1


def exat_pooled_fea(audi_or_vide_fea, video_mask, textgrid_path, phn_list, text_fea, text_mask, p2idx=p2idx, idx2p=idx2p):

    B, T, _ = audi_or_vide_fea.shape
    len_audi_or_vide = video_mask.sum(dim = 1)
    
    bath_list = []
    bath_text = []
    bath_emb = []
    mask = []
    for i in range(B):
        tg = textgrid.openTextgrid(textgrid_path[i], includeEmptyIntervals=True)
        maxtime = tg.maxTimestamp

        tier_name = "phones"
        phn_fea = []
        lab_p = []
        len_a = int(len_audi_or_vide[i])
        if tier_name in tg.tierNames:
            tier = tg._tierDict[tier_name]
            for interval in tier.entries:
                bgn_time, end_time, label = interval      
                if label == '' or label == 'sp' or label == 'spn' or label == 'sil' or label == 'noise':
                    continue
                else:
                    lab_p.append(p2idx[label])

                    bgn_embd_float = max(0, len_a * (bgn_time / maxtime))
                    end_embd_float = max(len_a * (end_time / maxtime), 1)

                    bgn_embd = math.floor(bgn_embd_float)
                    end_embd = math.ceil(end_embd_float)

                    if bgn_embd == end_embd:
                        if 1 - (bgn_embd_float % 1) >= end_embd_float % 1:
                            bgn_embd -= 1
                        else:
                            end_embd += 1

                    phone_fea = audi_or_vide_fea[i, bgn_embd:end_embd, :]

                    phn_fea.append(phone_fea)

            masktab = phn_list[i] != 71

            label_text = phn_list[i][phn_list[i] != 0]
            mask_t = torch.bitwise_and(masktab, text_mask[i].bool().squeeze())
            label_text = label_text[label_text!=71]
            common_indices_av, common_indices_t = find_common_indices(lab_p, list(label_text))
            pooled_av_fea = []
            for idx_av, idx_t in zip(common_indices_av, common_indices_t):
                cur_av_fea = phn_fea[idx_av]
                cur_t_fea = text_fea[i][idx_t]
            
                cur_pooled_av_fea = maxpool_with_gaussian(cur_av_fea, cur_t_fea).unsqueeze(0)
                pooled_av_fea.append(cur_pooled_av_fea)
            if len(pooled_av_fea) == 0:
                return 1
            phn_embd = torch.cat(pooled_av_fea, dim=0)

            bath_list.append(phn_embd)

            bath_text.append(label_text[common_indices_t])

            bath_emb.append(text_fea[i][common_indices_t, :])
            mask.append(mask_t[common_indices_t]) #embd mask

    if len(bath_list) != 0:
        audio_or_video_phnfea = torch.cat(bath_list, dim=0)
        phn_text = torch.cat(bath_text, dim=0)
        phn_mask = torch.cat(mask, dim=0)

        phn_text_embd = torch.cat(bath_emb, dim=0)

        # 去除mask为0部分的text embedding，shape [T, 128]
        phn_text_embd2 = torch.masked_select(phn_text_embd, phn_mask.bool().unsqueeze(-1)).unsqueeze(1).view(-1,128)

        return audio_or_video_phnfea, phn_text, phn_text_embd2
    else:
        return 1
         
# 去掉71的''

# len(lab_p) == len(lab_e) 去掉重复的 [53, 54, 5, 42, 57, 7, 42, 7, 43] [53, 54,  5, 42, 57, 35, 42,  7, 43,
# len(lab_p) < len(lab_e)  emb_t 减小 [44, 36, 55, 57, 26, 41, 2, 45] [44, 36, 55, 57, 26, 71, 41,  2, 45]
# len(lab_p) > len(lab_e)  emb_a 减小

# len != 

# ''' 35 practical
# [53, 54, 5, 42, 57, 7, 42, 7, 43]
# [44, 36, 55, 57, 26, 41, 2, 45]
# [42, 67, 2, 45, 7, 57, 38]
# [55, 18, 7, 45, 57, 36, 32, 35, 42]
# '''


# for tier_name in tg.tierNames:
#     tier = tg._tierDict[tier_name]
#     print(f"Tier name: {tier_name}")
#     for interval in tier.entries:
#         bgn_time, end_time, label = interval
#         print(f"  Interval: start={bgn_time}, end={end_time}, label={label}")


# len_a = 32
'''
# B, 100, 345
# B, P, 100, 345
maxpool, avgpool
(BP, 128)
# BP, 100, 345 mask
# BP, 100, 128

audio_features:  BP,128
text_features:  BP,128 (G2P)/train20/intern/permanent/kwli2/udkws/attakws/save_model_log/udv2_at_AWS2

# text phone = audio phone

1、切片 pool linear
2、linear 切片 pool
3、切片 linear attation pool

1、1,5,384 1,4,384  4,N,384   5,N,384 ->pool, linear ->K,128

B, T, 384 -> B, T, 128 -> B, T, 128 cross

cut的位置
audio encoder train
B, T, 384 (L)-> B, T, 128 cut-> BP, T, 128 pool-> BP, 1, 128 (L)-> BP, 128
B, T, 384 (L)-> B, T, 128 (L)-> B, T, 128 ->cross self_att -> out(0,1)

audio encoder interfence
B, T, 384 (L)-> B, T, 128 (L)-> B, T, 128 ->cross self_att 

out
K, 128 K,128

Loss

'''

