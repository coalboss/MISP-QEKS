

import torch
import torch.nn as nn
import torch.optim as optim


'''
Loss 参数: (audio_features  text_features) prob, (audio_phn, text_phn) mask
'''


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, audio_features, text_features):
        batch_size = audio_features.shape[0]
        audio_features = audio_features / audio_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logits = torch.matmul(audio_features, text_features.T) / self.temperature
        labels = torch.arange(batch_size).to(audio_features.device)
        loss_i2t = nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2

class ContrastiveLoss_mask(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss_mask, self).__init__()
        self.temperature = temperature
        
    def forward(self, audio_features, text_features, a_g2p, a_g2pmask=None):
        '''
        4, 128
        a_g2p mask phn
        audio_features, text_features,  phone_prob embedding
        a_g2p, a_g2pmask,  mask embedding
        '''
        batch_size = audio_features.shape[0]
        audio_features = audio_features / audio_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # logits = torch.matmul(audio_features, text_features.T) / self.temperature
        logits = torch.matmul(audio_features, text_features.T)

        labels = torch.arange(batch_size).to(audio_features.device)
        if a_g2pmask is not None:
            paired_label = torch.masked_select(a_g2p, a_g2pmask.bool().squeeze(-1)).unsqueeze(1)
        similarity_matrix = (a_g2p.unsqueeze(1) == a_g2p.unsqueeze(0))
        similarity_matrix.fill_diagonal_(False)
        logits[similarity_matrix] = -float('inf')

        loss_i2t = nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2



class ContrastiveLoss_mask_utt(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss_mask_utt, self).__init__()
        self.temperature = temperature
        
    def forward(self, audio_features, text_features, a_g2p, a_g2pmask=None):
        '''
        4, 128
        a_g2p mask phn
        audio_features, text_features,  phone_prob embedding
        a_g2p, a_g2pmask,  mask embedding
        '''
        batch_size = audio_features.shape[0]
        audio_features = audio_features / audio_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # logits = torch.matmul(audio_features, text_features.T) / self.temperature
        logits = torch.matmul(audio_features, text_features.T)

        labels = torch.arange(batch_size).to(audio_features.device)
        if a_g2pmask is not None:
            paired_label = torch.masked_select(a_g2p, a_g2pmask.bool().squeeze(-1)).unsqueeze(1)
        # similarity_matrix = (a_g2p.unsqueeze(1) == a_g2p.unsqueeze(0))
        similarity_matrix = (a_g2p.unsqueeze(1) == a_g2p.unsqueeze(0)).all(dim=-1)
        
        similarity_matrix.fill_diagonal_(False)
        logits[similarity_matrix] = -float('inf')


        loss_i2t = nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2


class ContrastiveLossword_mask(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLossword_mask, self).__init__()
        self.temperature = temperature
        
    def forward(self, audio_features, text_features, a_g2p, a_g2pmask=None):
        '''
        4, 128
        a_g2p mask phn
        audio_features, text_features,  phone_prob embedding
        a_g2p, a_g2pmask,  mask embedding
        '''
        batch_size = audio_features.shape[0]
        audio_features = audio_features / audio_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # logits = torch.matmul(audio_features, text_features.T) / self.temperature
        logits = torch.matmul(audio_features, text_features.T)

        labels = torch.arange(batch_size).to(audio_features.device)
        if a_g2pmask is not None:
            paired_label = torch.masked_select(a_g2p, a_g2pmask.bool().squeeze(-1)).unsqueeze(1)
        similarity_matrix = torch.tensor([[s1 == s2 for s2 in a_g2p] for s1 in a_g2p])
        similarity_matrix.fill_diagonal_(False)
        logits[similarity_matrix] = -float('inf')


        loss_i2t = nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2

