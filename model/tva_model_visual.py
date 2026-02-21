import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer, BertModel
from transformers import DebertaV2Tokenizer, AutoProcessor, WhisperProcessor

from model.fadata import *
from model.cosafind import *
from model.clap_phn import ContrastiveLoss_mask, ContrastiveLossword_mask, ContrastiveLoss_mask_utt
from model.transformer2 import Transformer_self, Transformer_cross, Transformer_encoder, CrossAttentionLayer, Transformer_encoder_pad, CrossModalAttention, NoiseReductionMask, MultiModalAdapter

from library.tools.creation import *

class Projection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Projection, self).__init__()
        layers = []
        layers.append(nn.LayerNorm(input_dim)) 
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Linear(output_dim, output_dim))
        layers.append(nn.SiLU())
        self.projection_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.projection_block(x)

class GRUFCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUFCModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_last_output = gru_out[:, -1, :]
        fc_out = self.fc(gru_last_output)
        return fc_out

class AudioKLDLoss(nn.Module):
    def __init__(self, temperature=1.0, reduction='batchmean'):
        super().__init__()
        self.kld = nn.KLDivLoss(reduction=reduction, log_target=False)
        self.temperature = temperature

    def forward(self, pred_emb, target_emb):

        p = F.softmax(pred_emb / self.temperature, dim=-1)
        q = F.softmax(target_emb / self.temperature, dim=-1)
        
        return self.kld(p.log(), q)

def gaussian_kernel(size, sigma):
        coords = torch.tensor([x - (size - 1) / 2 for x in range(size)], dtype=torch.float32)
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / torch.sum(kernel)
        return kernel

class TVA_KWS_PLCL_AVmask(nn.Module):
    def __init__(self, embedding = 128):
        super(TVA_KWS_PLCL_AVmask, self).__init__()

        # Text Projection
        self.Text_Proj = Projection(input_dim=256, output_dim=128)

        # Video Projection
        self.Vide_Proj = Projection(input_dim=256, output_dim=128)
        
        # Auido Projection
        self.Audi_ErlProj = Projection(input_dim=384, output_dim=128)
        self.Audi_QryProj = Projection(input_dim=384, output_dim=128)

        self.transformer_V_CAFE = CrossModalAttention()
        self.denoise_masking = NoiseReductionMask()

        self.mm_adapter = MultiModalAdapter(embed_dim=128, num_heads=1, dropout=0.0)
        # self.mm_adapter = Transformer_encoder(d_model = 128, nlayers = 2, nhead = 1, dim_feedforward = 512, dropout=0.1) # self attention

        # Multi modal feature substitution ratio
        self.ratio_replace = 0.5
        self.ratio_a2v     = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.ratio_v2a     = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Gussian kernel & Other Parameters
        self.kernel_size  = 5
        self.sigma        = 1.0
        self.alpha        = [0.75, 0.25]
        self.maxlen_text  = 40
        self.maxlen_vide  = 50
        self.maxlen_audi  = 100
        self.phonemes     = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', ' '] 
        self.p2idx        = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p        = {idx: p for idx, p in enumerate(self.phonemes)}
        self.rng          = np.random.default_rng(42)

        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        # Attention Layers
        self.transformer_aa = Transformer_cross(embed_size = self.maxlen_audi, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)
        self.transformer_ta = Transformer_cross(embed_size = self.maxlen_audi, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)
        self.transformer_tv = Transformer_cross(embed_size = self.maxlen_vide, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)
        self.transformer_av = Transformer_cross(embed_size = self.maxlen_vide, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)

        # MultiModal GRU-FC Layers
        self.gru_t_a = GRUFCModel(input_dim=self.maxlen_audi, hidden_dim=100, output_dim=100)
        self.gru_a_a = GRUFCModel(input_dim=self.maxlen_audi, hidden_dim=100, output_dim=100)
        self.gru_t_v = GRUFCModel(input_dim=self.maxlen_vide, hidden_dim=100, output_dim=100)
        self.gru_a_v = GRUFCModel(input_dim=self.maxlen_vide, hidden_dim=100, output_dim=100)

        # Classification Layers
        self.fc_t_a   = nn.Linear(100, 1)  # Text
        self.fc_t_v   = nn.Linear(100, 1)  # Text
        self.fc_a_a   = nn.Linear(100, 1)  # Video
        self.fc_a_v   = nn.Linear(100, 1)  # Audio
        self.fc_ta_a  = nn.Linear(200, 1)  # Text + Audio
        self.fc_ta_v  = nn.Linear(200, 1)  # Text + Audio
        self.fc_ta_va = nn.Linear(400, 1)  # Text + Video + Audio

        # Model criterions
        self.BCE_Loss    = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))  # sigmoid + BCEloss
        self.FCl_loss    = FocalLossweight(alpha=self.alpha, gamma=2)   # Focal Loss
        self.Contra_Loss = ContrastiveLoss_mask()   # Contrastive Loss
        self.MSE_Loss    = nn.MSELoss()
        self.KLD_Loss    = AudioKLDLoss()
    
    def forward(self, data, meta, fa_path, modality='ta_va'):

        claploss_tv = torch.tensor(0.0)
        claploss_ta = torch.tensor(0.0)
        claploss_av = torch.tensor(0.0)
        claploss_aa = torch.tensor(0.0)

        loss_t_a    = torch.tensor(0.0)
        loss_a_a    = torch.tensor(0.0)
        loss_ta_a   = torch.tensor(0.0)
        loss_ta_va  = torch.tensor(0.0)


        glm_loss = torch.tensor(0.0)

        assert modality in ['t_a', 't_v', 'a_v', 'a_a', 'ta_v', 'ta_a', 'ta_va'], " Unrecognized modality, support ['t_a', 't_v', 'a_v', 'a_a', 'ta_v', 'ta_a', 'ta_va']"

        anc_phn_list  = data['anc_phn_list']
        anc_text_fea  = data['anc_text_fea'] if modality in ['t_a', 't_v', 'ta_v', 'ta_a', 'ta_va'] else None
        anc_text_mask = data['anc_text_mask']
        anc_vide_fea  = data['anc_vide_fea']
        anc_vide_mask = data['anc_vide_mask']
        anc_audi_fea  = data['anc_audi_fea'] if modality in ['a_v', 'a_a', 'ta_v', 'ta_a', 'ta_va'] else None
        anc_audi_mask = data['anc_audi_mask']

        com_phn_list  = data['com_phn_list']
        com_text_fea  = data['com_text_fea']
        com_text_mask = data['com_text_mask']
        com_vide_fea  = data['com_vide_fea'] if modality in ['t_v', 'a_v', 'ta_v', 'ta_va'] else None
        com_vide_mask = data['com_vide_mask']
        com_audi_fea  = data['com_audi_fea'] if modality in ['t_a', 'a_v', 'a_a', 'ta_v', 'ta_a', 'ta_va'] else None
        com_audi_mask = data['com_audi_mask']

        anc_fa_path   = fa_path['anc_fa_path']
        com_fa_path   = fa_path['com_fa_path']
        
        label         = meta['label']
        B, _, _ = com_audi_fea.shape

        # Set None-speech segments to 0
        anc_audi_fea = anc_audi_fea * anc_audi_mask
        com_audi_fea = com_audi_fea * com_audi_mask




        # #############################################################################
        # ########################### Denoise Module Begin ############################
        # if modality in ['t_v', 'a_v', 'ta_v', 'ta_va']:
        #     denoised_audi_fea = []
        #     for i in range(B):
        #         visual_context = []

        #         len_audi = com_audi_mask[i].sum(dim=0)
        #         len_vide = com_vide_mask[i].sum(dim=0)
        #         f_audio = com_audi_fea[i, :int(len_audi), :]
        #         f_video = com_vide_fea[i, :int(len_vide), :]

        #         for t in range(int(len_audi)):
        #             visual_context.append(self.transformer_V_CAFE(f_audio[t, :], f_video))

        #         visual_context_batch = torch.stack(visual_context, dim=0)
        #         denoised_audio = self.denoise_masking(visual_context_batch, f_audio)
        #         denoised_audio = F.pad(denoised_audio, (0, 0, 0, self.maxlen_audi - denoised_audio.size(0)), "constant", 0)
        #         denoised_audi_fea.append(denoised_audio)

        #     com_audi_fea = torch.stack(denoised_audi_fea, dim=0)
        
        # if self.training:
        #     clean_com_audi_fea = data['clean_com_audi_fea'] * com_audi_mask
        #     denoised_loss = self.KLD_Loss(com_audi_fea, clean_com_audi_fea)
        # ############################ Denoise Module End #############################
        # #############################################################################



        #############################################################################
        ########################### Adapter Module Begin ############################

        # Text Enroll & Query
        if anc_text_fea != None:
            anc_text = self.Text_Proj(anc_text_fea) 
            com_text = self.Text_Proj(com_text_fea) 

        # Audio Enroll
        if anc_audi_fea != None:
            anc_audi = self.Audi_ErlProj(anc_audi_fea)

        # Video Enroll & Query
        if com_vide_fea != None:
            com_vide = self.Vide_Proj(com_vide_fea)
            anc_vide = self.Vide_Proj(anc_vide_fea) 

        # Audio Query
        if com_audi_fea != None:
            com_audi = self.Audi_QryProj(com_audi_fea)

        # Multi Modal Feature Substitution
        if self.training:
            lens_anc_vide, lens_anc_audi, lens_com_vide, lens_com_audi = anc_vide_mask.sum(dim=1), anc_audi_mask.sum(dim=1), com_vide_mask.sum(dim=1), com_audi_mask.sum(dim=1)
            gods_choice = self.rng.random(1)[0]
            if gods_choice > 0.5:       # Replacing Video with Audio
                new_anc_vide = anc_vide.clone()
                new_com_vide = com_vide.clone()
                glm_loss_anc = 0
                glm_loss_com = 0
                for b in range(B):
                    tmp_choice = self.rng.random(1)[0]
                    if tmp_choice > self.ratio_replace:
                        continue
                    ratio_v2a = self.rng.choice(self.ratio_v2a)
                    anc_replace_num = max(int(lens_anc_vide[b] * ratio_v2a), 1)
                    com_replace_num = max(int(lens_com_vide[b] * ratio_v2a), 1)

                    anc_replace_indices = self.rng.integers(low=0, high=int(lens_anc_vide[b])-1, size=anc_replace_num)
                    com_replace_indices = self.rng.integers(low=0, high=int(lens_com_vide[b])-1, size=com_replace_num)

                    for anc_idx in anc_replace_indices:
                        idx = min(torch.ceil( (anc_idx * lens_anc_audi[b]) / lens_anc_vide[b] ), self.maxlen_audi)
                        replacement_frame = anc_audi[b, int(idx)]
                        new_anc_vide[b, anc_idx] = replacement_frame

                    for com_idx in com_replace_indices:
                        idx = min(torch.ceil( (com_idx * lens_com_audi[b]) / lens_com_vide[b] ), self.maxlen_audi)
                        replacement_frame = com_audi[b, int(idx)]
                        new_com_vide[b, com_idx] = replacement_frame
                # Adapter
                new_anc_vide = self.mm_adapter(new_anc_vide)
                new_com_vide = self.mm_adapter(new_com_vide)

                # Collapse
                anc_fa_out = exat_pooled_fea(new_anc_vide, anc_vide_mask, anc_fa_path, anc_phn_list, anc_text, anc_text_mask)
                com_fa_out = exat_pooled_fea(new_com_vide, com_vide_mask, com_fa_path, com_phn_list, com_text, com_text_mask)  
                if anc_fa_out != 1:
                    new_anc_vide_phn_fea, new_anc_phn_list, new_anc_text_phn_fea = anc_fa_out
                    glm_loss_anc = self.MSE_Loss(new_anc_vide_phn_fea, new_anc_text_phn_fea)

                if com_fa_out != 1:
                    new_com_vide_phn_fea, new_com_phn_list, new_com_text_phn_fea = com_fa_out
                    glm_loss_com = self.MSE_Loss(new_com_vide_phn_fea, new_com_text_phn_fea)

            else:                       # Replacing Audio with Video
                new_anc_audi = anc_audi.clone()
                new_com_audi = com_audi.clone()
                glm_loss_anc = 0
                glm_loss_com = 0
                for b in range(B):
                    tmp_choice = self.rng.random(1)[0]
                    if tmp_choice > self.ratio_replace:
                        continue
                    ratio_a2v = self.rng.choice(self.ratio_a2v)
                    anc_replace_num = int(lens_anc_audi[b] * ratio_a2v)
                    com_replace_num = int(lens_com_audi[b] * ratio_a2v)

                    anc_replace_indices = self.rng.integers(low=0, high=int(lens_anc_audi[b]), size=anc_replace_num)
                    com_replace_indices = self.rng.integers(low=0, high=int(lens_com_audi[b]), size=com_replace_num)

                    for anc_idx in anc_replace_indices:
                        idx = min(torch.ceil( (anc_idx * lens_anc_vide[b]) / lens_anc_audi[b] ), self.maxlen_vide-1)
                        replacement_frame = anc_vide[b, int(idx)]
                        new_anc_audi[b, anc_idx] = replacement_frame

                    for com_idx in com_replace_indices:
                        idx = min(torch.ceil( (com_idx * lens_com_vide[b]) / lens_com_audi[b] ), self.maxlen_vide-1)
                        replacement_frame = com_vide[b, int(idx)]
                        new_com_audi[b, com_idx] = replacement_frame

                # Adapter
                new_anc_audi = self.mm_adapter(new_anc_audi)
                new_com_audi = self.mm_adapter(new_com_audi)

                anc_fa_out = exat_pooled_fea(new_anc_audi, anc_audi_mask, anc_fa_path, anc_phn_list, anc_text, anc_text_mask)
                com_fa_out = exat_pooled_fea(new_com_audi, com_audi_mask, com_fa_path, com_phn_list, com_text, com_text_mask)  
                if anc_fa_out != 1:
                    new_anc_audi_phn_fea, new_anc_phn_list, new_anc_text_phn_fea = anc_fa_out
                    glm_loss_anc = self.MSE_Loss(new_anc_audi_phn_fea, new_anc_text_phn_fea)

                if com_fa_out != 1:
                    new_com_audi_phn_fea, new_com_phn_list, new_com_text_phn_fea = com_fa_out
                    glm_loss_com = self.MSE_Loss(new_com_audi_phn_fea, new_com_text_phn_fea)

            glm_loss = glm_loss_anc + glm_loss_com


        # if anc_text_fea != None:
            # anc_text = self.mm_adapter(anc_text)

        if anc_audi_fea != None:
            anc_audi_aligned = self.mm_adapter(anc_audi)
        if com_audi_fea != None:
            com_audi_aligned = self.mm_adapter(com_audi)
        if com_vide_fea != None:
            com_vide_aligned = self.mm_adapter(com_vide)


        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d

        # 插值到相同的时间维度
        target_length = 100
        # text_features_interp = torch.nn.functional.interpolate(com_text.permute(0, 2, 1), size=target_length, mode='linear', align_corners=False).permute(0, 2, 1)
        # audio_features_interp = com_audi  # 音频已经是目标长度
        # video_features_interp = torch.nn.functional.interpolate(com_vide.permute(0, 2, 1), size=target_length, mode='linear', align_corners=False).permute(0, 2, 1)
        lens_anc_text, lens_anc_vide, lens_anc_audi, lens_com_text, lens_com_vide, lens_com_audi = anc_text_mask.sum(dim=1), anc_vide_mask.sum(dim=1), anc_audi_mask.sum(dim=1), com_text_mask.sum(dim=1), com_vide_mask.sum(dim=1), com_audi_mask.sum(dim=1)
        import pdb; pdb.set_trace()

        text_features_np  = com_text[:,:lens_com_text,:].squeeze().cpu().numpy()
        audio_features_np = com_audi[:,:lens_com_audi,:].squeeze().cpu().numpy()
        video_features_np = com_text[:,:lens_com_text,:].squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, (features, title) in enumerate(zip([text_features_np, audio_features_np, video_features_np], 
                                                ['Text Features', 'Audio Features', 'Video Features'])):
            heatmap = axes[i].imshow(features, aspect='auto', cmap='viridis')
            axes[i].set_title(title)
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Feature Dimensions')
            fig.colorbar(heatmap, ax=axes[i])

        # 保存为PDF
        plt.tight_layout()
        plt.savefig('heatmap_visualization.pdf', format='pdf')
        plt.show()

        # data_dict={
        #         'com_text':com_text,
        #         'com_audi':com_audi,
        #         'com_audi_aligned':com_audi_aligned, 
        #         'com_vide':com_vide, 
        #         'com_vide_aligned':com_vide_aligned, 
        #         }
        # # import pdb; pdb.set_trace()
        # npy_save = '/train8/asrkws/shicheng2/Data/Multimodal_KWS/LRS2/analysis/'
        # name = '_'.join(com_fa_path[0].split('/')[-2:]).split('.')[0]
        # save_path = npy_save + name + '.npy'

        # directory = os.path.dirname(save_path)
        # os.makedirs(directory, exist_ok=True)
        # np.save(save_path, data_dict)



        # ########################### Adapter Module Ended ############################
        # #############################################################################


        # ############################# PLCL Module Ended #############################

        # if self.training:

        #     ######################### Contrastive Learning between [ Enroll Text <----> Query Video ] ############################
        #     tv_ave_fea = tv_exatphone_pool(com_vide, com_vide_mask, com_fa_path, anc_phn_list, anc_text, anc_text_mask) 
        #     if tv_ave_fea != 1:
        #         com_phn_vide_embd_tv, tv_phn_list, anc_phn_text_embd_tv, length_1 = tv_ave_fea
        #         claploss_tv = self.Contra_Loss(com_phn_vide_embd_tv, anc_phn_text_embd_tv, tv_phn_list)

        #     ######################### Contrastive Learning between [ Enroll Text <----> Query Audio ] ############################
        #     ta_ave_fea = ta_exatphone_pool(com_audi, com_audi_mask, com_fa_path, anc_phn_list, anc_text, anc_text_mask) 
        #     if ta_ave_fea != 1:
        #         com_phn_audi_embd_ta, ta_phn_list, anc_phn_text_embd_ta, length_1 = ta_ave_fea
        #         claploss_ta = self.Contra_Loss(com_phn_audi_embd_ta, anc_phn_text_embd_ta, ta_phn_list)

        #     ######################### Contrastive Learning between [ Enroll Audio <----> Query Video ] ###########################
        #     av_ave_fea = va_exatphone_pool(anc_audi, anc_audi_mask, anc_fa_path, com_vide, com_vide_mask, com_fa_path, label) 
        #     if av_ave_fea != 1:
        #         anc_phn_audi_embd_av, av_phn_list, com_phn_vide_embd_av, length_1 = av_ave_fea
        #         claploss_av = self.Contra_Loss(anc_phn_audi_embd_av, com_phn_vide_embd_av, av_phn_list)

        #     ######################### Contrastive Learning between [ Enroll Audio <----> Query Audio ] ###########################
        #     aa_ave_fea = aa_exatphone_pool(anc_audi, anc_audi_mask, anc_fa_path, com_audi, com_audi_mask, com_fa_path, label) 
        #     if aa_ave_fea != 1:
        #         anc_phn_audi_embd_aa, aa_phn_list, com_phn_audi_embd_aa, length_1 = aa_ave_fea
        #         claploss_aa = self.Contra_Loss(anc_phn_audi_embd_aa, com_phn_audi_embd_aa, aa_phn_list)

        # ###############################################################################################################################
        # ##################################################| Later Fusion Module Begin |################################################
        # ###############################################################################################################################
        # aa_upsample_list = []
        # ta_upsample_list = []
        # tv_upsample_list = []
        # av_upsample_list = []

        # # A attention
        # if com_audi_fea != None:
        #     ######################### Attention between [ Enroll Text <----> Query Audio ] ###########################
        #     if anc_text_fea != None:
        #         ta_cosine_sim_matrix = calculate_cross(anc_text, com_audi, anc_text_mask, com_audi_mask)
        #         ta_cosine_sim_matrix = ta_cosine_sim_matrix.unsqueeze(1)
        #         for i in range(B):
        #             len_anc = anc_text_mask[i].sum()
        #             len_com = com_audi_mask[i].sum()
        #             cosine_sim_matrix_tmp = ta_cosine_sim_matrix[i, 0, :int(len_anc), :int(len_com)].unsqueeze(0).unsqueeze(1)
        #             upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(self.maxlen_text, self.maxlen_audi), mode='bilinear', align_corners=True)
        #             smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
        #             ta_upsample_list.append(smoothed_tensor)

        #         ta_anchor_speech = pad_sequence(ta_upsample_list, batch_first=True, padding_value=0.0).squeeze(1)
        #         ta_max_corss_speech = ta_anchor_speech.max(dim = 1).values.unsqueeze(1)
        #         pattern_ta = self.transformer_ta(ta_anchor_speech, ta_anchor_speech, ta_max_corss_speech)
        #         embd_t_a = self.gru_t_a(pattern_ta)

        #     ######################### Attention between [ Enroll Audio <----> Query Audio ] ###########################
        #     if anc_audi_fea != None:
        #         aa_cosine_sim_matrix = calculate_cross(anc_audi, com_audi, anc_audi_mask, com_audi_mask)
        #         aa_cosine_sim_matrix = aa_cosine_sim_matrix.unsqueeze(1)
        #         for i in range(B):
        #             len_anc = anc_audi_mask[i].sum()
        #             len_com = com_audi_mask[i].sum()
        #             cosine_sim_matrix_tmp = aa_cosine_sim_matrix[i, 0, :int(len_anc), :int(len_com)].unsqueeze(0).unsqueeze(1)
        #             upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(self.maxlen_audi, self.maxlen_audi), mode='bilinear', align_corners=True)
        #             smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
        #             aa_upsample_list.append(smoothed_tensor)

        #         aa_anchor_speech = pad_sequence(aa_upsample_list, batch_first=True, padding_value=0.0).squeeze(1)
        #         aa_max_corss_speech = aa_anchor_speech.max(dim = 1).values.unsqueeze(1)
        #         pattern_aa = self.transformer_aa(aa_anchor_speech, aa_anchor_speech, aa_max_corss_speech)
        #         embd_a_a = self.gru_a_a(pattern_aa)

        # # V attention
        # if com_vide_fea != None:
        #     ######################### Attention between [ Enroll Text <----> Query Video ] ###########################
        #     if anc_text_fea != None:
        #         tv_cosine_sim_matrix = calculate_cross(anc_text, com_vide, anc_text_mask, com_vide_mask)
        #         tv_cosine_sim_matrix = tv_cosine_sim_matrix.unsqueeze(1)
        #         for i in range(B):
        #             len_anc = anc_text_mask[i].sum()
        #             len_com = com_vide_mask[i].sum()
        #             cosine_sim_matrix_tmp = tv_cosine_sim_matrix[i, 0, :int(len_anc), :int(len_com)].unsqueeze(0).unsqueeze(1)
        #             upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(self.maxlen_text, self.maxlen_vide), mode='bilinear', align_corners=True)
        #             smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
        #             tv_upsample_list.append(smoothed_tensor)

        #         tv_anchor_speech = pad_sequence(tv_upsample_list, batch_first=True, padding_value=0.0).squeeze(1)
        #         tv_max_corss_speech = tv_anchor_speech.max(dim = 1).values.unsqueeze(1)
        #         pattern_tv = self.transformer_tv(tv_anchor_speech, tv_anchor_speech, tv_max_corss_speech)
        #         embd_t_v = self.gru_t_v(pattern_tv)

        #     ######################### Attention between [ Enroll Audio <----> Query Vide ] ###########################
        #     if anc_audi_fea != None:
        #         av_cosine_sim_matrix = calculate_cross(anc_audi, com_vide, anc_audi_mask, com_vide_mask)
        #         av_cosine_sim_matrix = av_cosine_sim_matrix.unsqueeze(1)
        #         for i in range(B):
        #             len_anc = anc_audi_mask[i].sum()
        #             len_com = com_vide_mask[i].sum()
        #             cosine_sim_matrix_tmp = av_cosine_sim_matrix[i, 0, :int(len_anc), :int(len_com)].unsqueeze(0).unsqueeze(1)
        #             upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(self.maxlen_audi, self.maxlen_vide), mode='bilinear', align_corners=True)
        #             smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
        #             av_upsample_list.append(smoothed_tensor)

        #         av_anchor_speech = pad_sequence(av_upsample_list, batch_first=True, padding_value=0.0).squeeze(1)
        #         av_max_corss_speech = av_anchor_speech.max(dim = 1).values.unsqueeze(1)
        #         pattern_av = self.transformer_av(av_anchor_speech, av_anchor_speech, av_max_corss_speech)
        #         embd_a_v = self.gru_a_v(pattern_av)

        # if self.training: 

        #     label = label.unsqueeze(-1).float()

        #     ## [ 注册：文本 <————> 测试：音频 ]
        #     post_t_a = self.fc_t_a(embd_t_a)
        #     loss_t_a = self.BCE_Loss(post_t_a, label)

        #     ## [ 注册：音频 <————> 测试：音频 ]
        #     post_a_a = self.fc_a_a(embd_a_a)
        #     loss_a_a = self.BCE_Loss(post_a_a, label)

        #     # ## [ 注册：文本 <————> 测试：视频 ]
        #     # post_t_v = self.fc_t_v(embd_t_v)
        #     # loss_t_v = self.BCE_Loss(post_t_v, label)

        #     # ## [ 注册：音频 <————> 测试：视频 ]
        #     # post_a_v = self.fc_a_v(embd_a_v)
        #     # loss_a_v = self.BCE_Loss(post_a_v, label)

        #     ## [ 注册：文本+音频 <————> 测试：音频 ]
        #     len_ta_a = min(len(embd_t_a), len(embd_a_a))
        #     embd_ta_a = torch.cat((embd_t_a[:len_ta_a], embd_a_a[:len_ta_a]), dim = 1)    # TA
        #     post_ta_a = self.fc_ta_a(embd_ta_a)
        #     loss_ta_a = self.BCE_Loss(post_ta_a, label)

        #     # ## [ 注册：文本+音频 <————> 测试：视频 ]
        #     # len_ta_v = min(len(embd_t_v), len(embd_a_v))
        #     # embd_ta_v = torch.cat((embd_t_v[:len_ta_v], embd_a_v[:len_ta_v]), dim = 1)    # VA
        #     # post_ta_v = self.fc_ta_v(embd_ta_v)
        #     # loss_ta_v = self.BCE_Loss(post_ta_v, label)
            
        #     ## [ 注册：文本+音频 <————> 测试：音频+视频 ]
        #     len_ta_va = min(len(embd_t_v), len(embd_a_v), len(embd_t_a), len(embd_a_a))
        #     embd_ta_va = torch.cat((embd_t_a[:len_ta_va], embd_a_a[:len_ta_va], embd_t_v[:len_ta_va], embd_a_v[:len_ta_va]), dim = 1)
        #     post_ta_va = self.fc_ta_va(embd_ta_va)
        #     loss_ta_va = self.BCE_Loss(post_ta_va, label)

        #     out = post_ta_va

        # else:
        #     if com_audi_fea != None and com_vide_fea == None:
        #         if anc_text_fea != None and anc_audi_fea == None:               ## [ 注册：文本 <————> 测试：音频 ]
        #             out = self.fc_t_a(embd_t_a)
        #         if anc_text_fea == None and anc_audi_fea != None:               ## [ 注册：音频 <————> 测试：音频 ]
        #             out = self.fc_a_a(embd_a_a)
        #         if anc_text_fea != None and anc_audi_fea != None:               ## [ 注册：文本+音频 <————> 测试：音频 ]
        #             len_ta_a = min(len(embd_t_a), len(embd_a_a))
        #             embd_ta_a = torch.cat((embd_t_a[:len_ta_a], embd_a_a[:len_ta_a]), dim = 1) 
        #             out = self.fc_ta_a(embd_ta_a)

        #     # if com_audi_fea == None and com_vide_fea != None:
        #     #     if anc_text_fea != None and anc_audi_fea == None:               ## [ 注册：文本 <————> 测试：视频 ]
        #     #         post_t_v = self.fc_t_v(embd_t_v)
        #     #         out = post_t_v
        #     #     if anc_text_fea == None and anc_audi_fea != None:               ## [ 注册：音频 <————> 测试：视频 ]
        #     #         post_a_v = self.fc_a_v(embd_a_v)
        #     #         out = post_a_v
        #     #     if anc_text_fea != None and anc_audi_fea != None:               ## [ 注册：文本+音频 <————> 测试：视频 ]
        #     #         len_ta_v = min(len(embd_t_v), len(embd_a_v))
        #     #         embd_ta_v = torch.cat((embd_t_v[:len_ta_v], embd_a_v[:len_ta_v]), dim = 1) 
        #     #         post_ta_v = self.fc_ta_v(embd_ta_a)
        #     #         out = post_ta_v

        #     if com_audi_fea != None and com_vide_fea != None:
        #         if anc_text_fea != None and anc_audi_fea != None:               ## [ 注册：文本+音频 <————> 测试：音频+视频 ]
        #             len_ta_va = min(len(embd_t_v), len(embd_a_v), len(embd_t_a), len(embd_a_a))
        #             embd_ta_va = torch.cat((embd_t_a[:len_ta_va], embd_a_a[:len_ta_va], embd_t_v[:len_ta_va], embd_a_v[:len_ta_va]), dim = 1)
        #             out = self.fc_ta_va(embd_ta_va)

        # return out, claploss_tv, claploss_ta, claploss_av, claploss_aa, loss_t_a, loss_a_a, loss_ta_a, loss_ta_va, glm_loss
