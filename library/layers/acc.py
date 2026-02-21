from typing import Sequence
import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

class KWSIflytekAcc(jit.ScriptModule):
    def __init__(self):
        super(KWSIflytekAcc, self).__init__()

    def forward(self, logit, frames_label, num_classes, nmod, low_th, up_th, neg_id):
        frames_label = frames_label.reshape(-1, 1)
        logit = logit.reshape(-1, logit.size(-1))
        
        lprobs_pos = F.log_softmax(logit, dim=1)
        lprobs_neg = lprobs_pos.clone()
        
        # positive acc
        target = frames_label.clone().flatten()
        target[target<=low_th] = -1
        target[target>=up_th] = -1
        target = target.long().reshape(-1, nmod)[:,0]
        target = target.unsqueeze(-1)
        _, new_target = torch.broadcast_tensors(lprobs_pos, target)
        remove_pad_mask = new_target.ne(-1)
        
        lprobs_pos = lprobs_pos[remove_pad_mask]
        lprobs_pos = lprobs_pos.reshape((-1, num_classes))
        if len(lprobs_pos) != 0:
            target = target[target!=-1]
            preds = torch.argmax(lprobs_pos, dim=1)
            correct_holder_pos = torch.eq(preds.squeeze(), target.squeeze()).float()

            num_corr_pos = int(correct_holder_pos.sum())
            num_sample_pos = int(torch.numel(correct_holder_pos))
            acc_pos = float(num_corr_pos)/float(num_sample_pos)
        else:
            acc_pos = -1

        # negtive acc
        target = frames_label.clone().flatten()
        target[target>neg_id] = -1
        target[target<neg_id] = -1
        target = target.long().reshape(-1, nmod)[:,0]
        target = target.unsqueeze(-1)
        _, new_target = torch.broadcast_tensors(lprobs_neg, target)
        remove_pad_mask = new_target.ne(-1)

        lprobs_neg = lprobs_neg[remove_pad_mask]
        lprobs_neg = lprobs_neg.reshape((-1, num_classes))
        if len(lprobs_neg) != 0:
            target = target[target!=-1]
            preds = torch.argmax(lprobs_neg, dim=1)
            preds[preds >= 9000] = 9000
            # print(preds.squeeze(),target.squeeze())
            correct_holder_neg = torch.eq(preds.squeeze(), target.squeeze()).float()

            num_corr_neg = int(correct_holder_neg.sum())
            num_sample_neg = int(torch.numel(correct_holder_neg))
            acc_neg = float(num_corr_neg)/float(num_sample_neg)
        else:
            acc_neg = -1
        return acc_pos, acc_neg

def WER(labs, recs):
    n_lab = len(labs)
    n_rec = len(recs)
    dist_mat = np.zeros((n_lab+1, n_rec+1))
    for j in range(n_rec + 1):
        dist_mat[0, j] = j
    
    for i in range(n_lab + 1):
        dist_mat[i, 0] = i

    for i in range(1, n_lab+1):
        for j in range(1, n_rec+1):
            hit_score = dist_mat[i-1,j-1] + (labs[i-1]!=recs[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1,j] + 1

            err = hit_score
            if err > ins_score:
                err = ins_score
            if err > del_score:
                err = del_score
            dist_mat[i, j]=err

    return dist_mat[n_lab, n_rec]

class ACC(jit.ScriptModule):
    def __init__(self):
        super(ACC, self).__init__()

    #@jit.script_method
    def forward(self, logit, frames_label, num_classes, nmod):
        lprobs = F.log_softmax(logit, dim=1)
        target = frames_label.clone()
        target = target.flatten()
        target[target<0] = -1
        target = target.long()
        target = target.reshape(-1, nmod)[:,0]
        target = target.unsqueeze(-1)
        _, new_target = torch.broadcast_tensors(lprobs, target)

        remove_pad_mask = new_target.ne(-1)
        lprobs = lprobs[remove_pad_mask]
        lprobs = lprobs.reshape((-1, num_classes))

        target = target[target!=-1]

        preds = torch.argmax(lprobs, dim=1)
        correct_holder = torch.eq(preds.squeeze(), target.squeeze()).float()

        num_corr = int(correct_holder.sum())
        num_sample = int(torch.numel(correct_holder))
        acc = float(num_corr)/float(num_sample)
        return acc

class CTC_WER(nn.Module):
    def __init__(self):
        super(CTC_WER, self).__init__()

    #@jit.script_method
    def forward(self, logit, ctc_list, blankID):
        lprobs = F.log_softmax(logit.squeeze(), dim=1) #t b numclasses
        _, logmax_index = torch.max(lprobs, 1)
        logmax_index = logmax_index[logmax_index!=blankID]

        ctc_pred = torch.from_numpy(np.array([]))
        for value in logmax_index:
                if len(ctc_pred)==0:
                    ctc_pred = value.unsqueeze(0)
                else:
                    if value != ctc_pred[-1]:
                        ctc_pred = torch.cat((ctc_pred, value.unsqueeze(0)), 0)

        ctc_list = ctc_list.squeeze(0)
        edist = self.Levenshtein_Distance(ctc_list, ctc_pred)
        wer = round(edist/len(ctc_list), 8)
        accuracy = 1 - wer
        return accuracy

    def Levenshtein_Distance(self, str1, str2):
        matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
        for i in range(1, len(str1)+1):
            for j in range(1, len(str2)+1):
                if(str1[i-1] == str2[j-1]):
                    d = 0
                else:
                    d = 1
                matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)
        return matrix[len(str1)][len(str2)]

class GetPostOut(jit.ScriptModule):
    def __init__(self, numclasses, nmode):
        super(GetPostOut, self).__init__()
        self.numclasses = numclasses
        self.nmode = nmode

    #@jit.script_method-> torch.Tensor
    def forward(self, x: torch.Tensor, label: torch.Tensor) :
        target = label.clone()
        target = target.flatten()
        target[target<0] = -1
        target = target.long()
        target = target.reshape(-1,self.nmode)[:,0]
        target = target.unsqueeze(-1)
        _, new_target = torch.broadcast_tensors(x, target)
        remove_pad_mask = new_target.ne(-1)
        x_mask = x[remove_pad_mask]
        x_mask = x_mask.reshape((-1, self.numclasses))
        return x_mask
