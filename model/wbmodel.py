import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append("/train20/intern/permanent/kwli2/udkws/attakws")
from transformer2 import Transformer_self, Transformer_cross, Transformer_encoder, Transformer_encoder_pad
from model.cosafind import *
from torch.nn.utils.rnn import pad_sequence

from model.fadata import *
from model.clap_phn import ContrastiveLoss_mask, ContrastiveLossword_mask

class Projection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Projection, self).__init__()
        layers = []
        layers.append(nn.LayerNorm(input_dim)) 
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.SiLU())
        self.projection_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.projection_block(x)

class speech_CPEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2):
        super(speech_CPEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class text_CPEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_kernel_size=2, pool_stride=2):
        super(text_CPEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class BERT_encoder(nn.Module):
    def __init__(self,):
        super(BERT_encoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('/train20/intern/permanent/kwli2/udkws/tmp1/bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('/train20/intern/permanent/kwli2/udkws/tmp1/bert-base-uncased')
        self.bert_model = self.bert_model.cuda()
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=10, padding='max_length', truncation=True)
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        # inputs = inputs.cuda()
        mask = inputs['attention_mask']
        outputs = self.bert_model(**inputs)
        x = outputs.last_hidden_state
        return x, mask

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

class discriminator_c(nn.Module):
    def __init__(self, input_size = 128, num_channels = 128):
        super(discriminator_c, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = input_size, out_channels = num_channels, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels = num_channels, out_channels = num_channels, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 407, 128) -> (B, 128, 407)
        x = self.conv1(x)      # (B, 128, 407) -> (B, 128, 407)
        x = F.relu(x)
        x = self.pool1(x)      # (B, 128, 407) -> (B, 128, 203)
        x = self.conv2(x)      # (B, 128, 203) -> (B, 128, 203)
        x = F.relu(x)
        x = self.pool2(x)      # (B, 128, 203) -> (B, 128, 101)
        x = self.global_pool(x)# (B, 128, 101) -> (B, 128, 1)
        x = x.squeeze(-1)      # (B, 128, 1) -> (B, 128)
        return x


class udkwsa(nn.Module):
    def __init__(self, embedding = 128):
        super(udkwsa, self).__init__()
        self.bert_encoder = BERT_encoder() 
        self.text_adp = text_CPEncoder(in_channels=768, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)
        self.audio_adp = speech_CPEncoder(in_channels=384, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)

        # self.attn = nn.MultiheadAttention(embed_dim=embedding, num_heads=1)

        self.transformer = Transformer_m(embed_size = 128, num_layers = 1, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
    
    def forward(self, anco_speech, anco_text, com_speech):
        anco_text_e = self.bert_encoder(anco_text) 
        anco_text_e = anco_text_e.cuda()

        anco_text_e = anco_text_e.transpose(1, 2)
        anco_text_p = self.text_adp(anco_text_e)

        anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp(anco_speech)

        com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp(com_speech)

        anco_text_p = anco_text_p.transpose(1, 2)
        anco_speech_p = anco_speech_p.transpose(1, 2)
        com_speech_p = com_speech_p.transpose(1, 2)


        pattern_p = self.transformer(com_speech_p, com_speech_p, anco_speech_p, mask = None)

        # speech_text = torch.cat((anco_speech_p, anco_text_p), dim=1)
        # pattern_p = self.transformer(speech_text, mask = None)  #(B,407,128)

        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        out = self.dis(pattern_p)
        out = self.dense(out)


        return out


class udkws_aa(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aa, self).__init__()
        self.bert_encoder = BERT_encoder() 
        self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)
        # self.text_adp = text_CPEncoder(in_channels=768, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)
        # self.audio_adp = speech_CPEncoder(in_channels=384, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)

        # self.attn = nn.MultiheadAttention(embed_dim=embedding, num_heads=1)

        self.transformer = Transformer_cross(embed_size = 128, num_layers = 1, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        self.transformer_e = Transformer_self(embed_size = 128, num_layers = 4, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        self.transformer_en = Transformer_encoder()

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        self.gru = GRUFCModel(input_dim=128, hidden_dim=128, output_dim=128)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        anco_text_e, mask_text = self.bert_encoder(anco_text) 
        anco_text_e = anco_text_e.cuda()
        cross = calculate_cross(anco_text_e,anco_text_e)
        cross = cross.cpu().detach().numpy()
        # anco_text_e = anco_text_e.transpose(1, 2)
        anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp(anco_speech)

        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp(com_speech)

        # anco_text_p = anco_text_p.transpose(1, 2)
        # anco_speech_p = anco_speech_p.transpose(1, 2)
        # com_speech_p = com_speech_p.transpose(1, 2)


        # pattern_p = self.transformer(com_speech_p, com_speech_p, anco_speech_p, mask = None)
        # print(anchor_mask.shape, comparison_mask.transpose(1, 2).shape)
        # mask_cross = torch.bmm(mask_text.unsqueeze(-1).float(),comparison_mask.transpose(1, 2))
        speech_s2_mask = torch.cat((anchor_mask, comparison_mask), dim=1)
        
        mask_cross = torch.bmm(speech_s2_mask,speech_s2_mask.transpose(1, 2))

        speech_text = torch.cat((anco_speech_p, com_speech_p), dim=1)
        pattern_p = self.transformer_en(speech_text, mask_cross)

        # speech_text = torch.cat((anco_speech_p, anco_text_p), dim=1)
        # pattern_p = self.transformer(speech_text, mask = None)  #(B,407,128)

        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        out = self.gru(pattern_p)
        out = self.dense(out)

        return out


class udkws_aa2(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aa2, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)
        # self.text_adp = text_CPEncoder(in_channels=768, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)
        # self.audio_adp = speech_CPEncoder(in_channels=384, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)

        # self.attn = nn.MultiheadAttention(embed_dim=embedding, num_heads=1)

        # self.transformer = Transformer_cross(embed_size = 128, num_layers = 1, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        # self.transformer_e = Transformer_self(embed_size = 128, num_layers = 4, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        self.transformer_en = Transformer_encoder()

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        self.gru = GRUFCModel(input_dim=128, hidden_dim=128, output_dim=128)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''

        anco_speech_p = self.audio_adp(anco_speech)
        com_speech_p = self.audio_adp(com_speech)

        speech_s2_mask = torch.cat((anchor_mask, comparison_mask), dim=1)
        
        mask_cross = torch.bmm(speech_s2_mask,speech_s2_mask.transpose(1, 2)).bool()
        # inf_tensor = torch.tensor(float('-inf')).cuda()
        # mask_cross = ~mask_cross.bool()
        # mask_cross = torch.masked_fill(mask_cross.float(), mask_cross.bool(), inf_tensor).float()

        speech_text = torch.cat((anco_speech_p, com_speech_p), dim=1)
        pattern_p = self.transformer_en(speech_text, mask_cross)

        out = self.gru(pattern_p)
        out = self.dense(out)


        return out

def gaussian_kernel(size, sigma):
        coords = torch.tensor([x - (size - 1) / 2 for x in range(size)], dtype=torch.float32)
        x, y = torch.meshgrid(coords, coords)
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / torch.sum(kernel)
        return kernel
        

class udkws_aatest(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aatest, self).__init__()
        self.bert_encoder = BERT_encoder() 
        self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)
        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()
        # self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 3), stride=(2, 1), padding=(1, 1), output_padding=(8, 0))
        # self.upsample = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=True)
        # self.text_adp = text_CPEncoder(in_channels=768, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)
        # self.audio_adp = speech_CPEncoder(in_channels=384, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)

        # self.attn = nn.MultiheadAttention(embed_dim=embedding, num_heads=1)

        # self.transformer = Transformer_cross(embed_size = 128, num_layers = 1, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        # self.transformer_e = Transformer_self(embed_size = 128, num_layers = 4, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        self.transformer_en = Transformer_encoder_pad(d_model = 100)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        anco_text_e, mask_text = self.bert_encoder(anco_text) 
        anco_text_e = anco_text_e.cuda()
        cross = calculate_cross(anco_text_e,anco_text_e)
        cross = cross.cpu().detach().numpy()
        # anco_text_e = anco_text_e.transpose(1, 2)
        anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp(anco_speech)
        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp(com_speech)
        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(anco_speech.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4
    # 16,100,100

        # anco_text_p = anco_text_p.transpose(1, 2)
        # anco_speech_p = anco_speech_p.transpose(1, 2)
        # com_speech_p = com_speech_p.transpose(1, 2)


        # pattern_p = self.transformer(com_speech_p, com_speech_p, anco_speech_p, mask = None)
        # print(anchor_mask.shape, comparison_mask.transpose(1, 2).shape)
        # mask_cross = torch.bmm(mask_text.unsqueeze(-1).float(),comparison_mask.transpose(1, 2))
        # speech_s2_mask = torch.cat((anchor_mask, comparison_mask), dim=1)
        
        # mask_cross = torch.bmm(speech_s2_mask,speech_s2_mask.transpose(1, 2))

        # speech_text = torch.cat((anco_speech_p, com_speech_p), dim=1)
        # pattern_p = self.transformer_en(pad_anchor_speech, comparison_mask)
        pattern_p = self.transformer_en(pad_anchor_speech)


        # speech_text = torch.cat((anco_speech_p, anco_text_p), dim=1)
        # pattern_p = self.transformer(speech_text, mask = None)  #(B,407,128)

        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        out = self.gru(pattern_p)
        out = self.dense(out)
        return out


class udkws_aa2proc(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aa2proc, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()
        # self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 3), stride=(2, 1), padding=(1, 1), output_padding=(8, 0))
        # self.upsample = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=True)
        # self.text_adp = text_CPEncoder(in_channels=768, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)
        # self.audio_adp = speech_CPEncoder(in_channels=384, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)

        # self.attn = nn.MultiheadAttention(embed_dim=embedding, num_heads=1)

        # self.transformer = Transformer_cross(embed_size = 128, num_layers = 1, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        # self.transformer_e = Transformer_self(embed_size = 128, num_layers = 4, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        self.transformer_en = Transformer_encoder_pad(d_model = 100)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        # anco_text_e, mask_text = self.bert_encoder(anco_text) 
        # anco_text_e = anco_text_e.cuda()
        # cross = calculate_cross(anco_text_e,anco_text_e)
        # cross = cross.cpu().detach().numpy()
        # anco_text_e = anco_text_e.transpose(1, 2)
        # anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp_an(anco_speech)
        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp_com(com_speech)
        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(anco_speech.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4
    # 16,100,100

        # anco_text_p = anco_text_p.transpose(1, 2)
        # anco_speech_p = anco_speech_p.transpose(1, 2)
        # com_speech_p = com_speech_p.transpose(1, 2)


        # pattern_p = self.transformer(com_speech_p, com_speech_p, anco_speech_p, mask = None)
        # print(anchor_mask.shape, comparison_mask.transpose(1, 2).shape)
        # mask_cross = torch.bmm(mask_text.unsqueeze(-1).float(),comparison_mask.transpose(1, 2))
        # speech_s2_mask = torch.cat((anchor_mask, comparison_mask), dim=1)
        
        # mask_cross = torch.bmm(speech_s2_mask,speech_s2_mask.transpose(1, 2))

        # speech_text = torch.cat((anco_speech_p, com_speech_p), dim=1)
        # pattern_p = self.transformer_en(pad_anchor_speech, comparison_mask)
        pattern_p = self.transformer_en(pad_anchor_speech)


        # speech_text = torch.cat((anco_speech_p, anco_text_p), dim=1)
        # pattern_p = self.transformer(speech_text, mask = None)  #(B,407,128)

        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        out = self.gru(pattern_p)
        out = self.dense(out)
        return out


class udkws_aa4proc(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aa4proc, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)
        self.audio_adp_an2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp_com2 = Projection(input_dim=128, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()
        # self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 3), stride=(2, 1), padding=(1, 1), output_padding=(8, 0))
        # self.upsample = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=True)
        # self.text_adp = text_CPEncoder(in_channels=768, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)
        # self.audio_adp = speech_CPEncoder(in_channels=384, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)

        # self.attn = nn.MultiheadAttention(embed_dim=embedding, num_heads=1)

        # self.transformer = Transformer_cross(embed_size = 128, num_layers = 1, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        # self.transformer_e = Transformer_self(embed_size = 128, num_layers = 4, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        self.transformer_en = Transformer_encoder_pad(d_model = 100)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        # anco_text_e, mask_text = self.bert_encoder(anco_text) 
        # anco_text_e = anco_text_e.cuda()
        # cross = calculate_cross(anco_text_e,anco_text_e)
        # cross = cross.cpu().detach().numpy()
        # anco_text_e = anco_text_e.transpose(1, 2)
        # anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp_an(anco_speech)
        anco_speech_p = self.audio_adp_an2(anco_speech_p)

        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp_com(com_speech)
        com_speech_p = self.audio_adp_com2(com_speech_p)

        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(anco_speech.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4
    # 16,100,100

        # anco_text_p = anco_text_p.transpose(1, 2)
        # anco_speech_p = anco_speech_p.transpose(1, 2)
        # com_speech_p = com_speech_p.transpose(1, 2)


        # pattern_p = self.transformer(com_speech_p, com_speech_p, anco_speech_p, mask = None)
        # print(anchor_mask.shape, comparison_mask.transpose(1, 2).shape)
        # mask_cross = torch.bmm(mask_text.unsqueeze(-1).float(),comparison_mask.transpose(1, 2))
        # speech_s2_mask = torch.cat((anchor_mask, comparison_mask), dim=1)
        
        # mask_cross = torch.bmm(speech_s2_mask,speech_s2_mask.transpose(1, 2))

        # speech_text = torch.cat((anco_speech_p, com_speech_p), dim=1)
        # pattern_p = self.transformer_en(pad_anchor_speech, comparison_mask)
        pattern_p = self.transformer_en(pad_anchor_speech)


        # speech_text = torch.cat((anco_speech_p, anco_text_p), dim=1)
        # pattern_p = self.transformer(speech_text, mask = None)  #(B,407,128)

        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        out = self.gru(pattern_p)
        out = self.dense(out)
        return out, out



class udkws_aacorss(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aacorss, self).__init__()
        self.bert_encoder = BERT_encoder() 
        self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)
        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()
        # self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 3), stride=(2, 1), padding=(1, 1), output_padding=(8, 0))
        # self.upsample = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=True)
        # self.text_adp = text_CPEncoder(in_channels=768, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)
        # self.audio_adp = speech_CPEncoder(in_channels=384, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)

        # self.attn = nn.MultiheadAttention(embed_dim=embedding, num_heads=1)

        self.transformer = Transformer_cross(embed_size = 100, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)
        # self.transformer_e = Transformer_self(embed_size = 128, num_layers = 4, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        self.transformer_en = Transformer_encoder_pad(d_model = 100)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, an_path, com_path, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        anco_text_e, mask_text = self.bert_encoder(anco_text) 
        anco_text_e = anco_text_e.cuda()
        cross = calculate_cross(anco_text_e,anco_text_e)
        cross = cross.cpu().detach().numpy()
        # anco_text_e = anco_text_e.transpose(1, 2)
        anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp(anco_speech)
        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp(com_speech)
        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(anco_speech.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4
        # 16,100,100
        # anco_text_p = anco_text_p.transpose(1, 2)
        # anco_speech_p = anco_speech_p.transpose(1, 2)
        # com_speech_p = com_speech_p.transpose(1, 2)


        # pattern_p = self.transformer(com_speech_p, com_speech_p, anco_speech_p, mask = None)
        # print(anchor_mask.shape, comparison_mask.transpose(1, 2).shape)
        # mask_cross = torch.bmm(mask_text.unsqueeze(-1).float(),comparison_mask.transpose(1, 2))
        # speech_s2_mask = torch.cat((anchor_mask, comparison_mask), dim=1)
        
        # mask_cross = torch.bmm(speech_s2_mask,speech_s2_mask.transpose(1, 2))

        # speech_text = torch.cat((anco_speech_p, com_speech_p), dim=1)
        # pattern_p = self.transformer_en(pad_anchor_speech, comparison_mask)
        # pattern_p = self.transformer_en(pad_anchor_speech)

        # max_corss_speech = pad_anchor_speech.mean(dim = 1).unsqueeze(1)
        max_corss_speech = pad_anchor_speech.max(dim = 1).values.unsqueeze(1)


        pattern_p = self.transformer(pad_anchor_speech, pad_anchor_speech, max_corss_speech)


        # speech_text = torch.cat((anco_speech_p, anco_text_p), dim=1)
        # pattern_p = self.transformer(speech_text, mask = None)  #(B,407,128)

        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        # out = self.gru(pattern_p)
        out = pattern_p.squeeze(1)
        out = self.dense(out)
        return out, out


class udkws_aacorss2(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aacorss2, self).__init__()
        self.bert_encoder = BERT_encoder() 
        self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)
        self.audio_adp_an2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp_com2 = Projection(input_dim=128, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()
        # self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 3), stride=(2, 1), padding=(1, 1), output_padding=(8, 0))
        # self.upsample = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=True)
        # self.text_adp = text_CPEncoder(in_channels=768, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)
        # self.audio_adp = speech_CPEncoder(in_channels=384, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)

        # self.attn = nn.MultiheadAttention(embed_dim=embedding, num_heads=1)

        self.transformer = Transformer_cross(embed_size = 100, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)
        self.transformer2 = Transformer_cross(embed_size = 100, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)

        # self.transformer_e = Transformer_self(embed_size = 128, num_layers = 4, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        self.transformer_en = Transformer_encoder_pad(d_model = 100)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(200, 1)
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, an_path, com_path, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        # anco_text_e, mask_text = self.bert_encoder(anco_text) 
        # anco_text_e = anco_text_e.cuda()
        # cross = calculate_cross(anco_text_e,anco_text_e)
        # cross = cross.cpu().detach().numpy()
        # anco_text_e = anco_text_e.transpose(1, 2)
        # anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp_an(anco_speech)
        anco_speech_p = self.audio_adp_an2(anco_speech_p)

        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp_com(com_speech)
        com_speech_p = self.audio_adp_com2(com_speech_p)

        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(anco_speech.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4
        # 16,100,100
        # anco_text_p = anco_text_p.transpose(1, 2)
        # anco_speech_p = anco_speech_p.transpose(1, 2)
        # com_speech_p = com_speech_p.transpose(1, 2)


        # pattern_p = self.transformer(com_speech_p, com_speech_p, anco_speech_p, mask = None)
        # print(anchor_mask.shape, comparison_mask.transpose(1, 2).shape)
        # mask_cross = torch.bmm(mask_text.unsqueeze(-1).float(),comparison_mask.transpose(1, 2))
        # speech_s2_mask = torch.cat((anchor_mask, comparison_mask), dim=1)
        
        # mask_cross = torch.bmm(speech_s2_mask,speech_s2_mask.transpose(1, 2))

        # speech_text = torch.cat((anco_speech_p, com_speech_p), dim=1)
        # pattern_p = self.transformer_en(pad_anchor_speech, comparison_mask)
        # pattern_p = self.transformer_en(pad_anchor_speech)

        # max_corss_speech = pad_anchor_speech.mean(dim = 1).unsqueeze(1)
        max_corss_anspeech = pad_anchor_speech.max(dim = 1).values.unsqueeze(1)
        pattern_pan = self.transformer(pad_anchor_speech, pad_anchor_speech, max_corss_anspeech)

        max_corss_comspeech = pad_anchor_speech.max(dim = 2).values.unsqueeze(1)
        pattern_pcom = self.transformer2(pad_anchor_speech.transpose(1,2), pad_anchor_speech.transpose(1,2), max_corss_comspeech)

        pattern_pan = torch.cat((pattern_pan, pattern_pcom), dim = 2)


        # speech_text = torch.cat((anco_speech_p, anco_text_p), dim=1)
        # pattern_p = self.transformer(speech_text, mask = None)  #(B,407,128)

        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        # out = self.gru(pattern_p)
        out = pattern_pan.squeeze(1)
        out = self.dense(out)
        return out, out


class udkws_aa2clapselfatt(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aa2clapselfatt, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)
        self.audio_adp_an2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp_com2 = Projection(input_dim=128, output_dim=128)
        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()
        # self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 3), stride=(2, 1), padding=(1, 1), output_padding=(8, 0))
        # self.upsample = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=True)
        # self.text_adp = text_CPEncoder(in_channels=768, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)
        # self.audio_adp = speech_CPEncoder(in_channels=384, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)

        # self.attn = nn.MultiheadAttention(embed_dim=embedding, num_heads=1)

        # self.transformer = Transformer_cross(embed_size = 128, num_layers = 1, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        # self.transformer_e = Transformer_self(embed_size = 128, num_layers = 4, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        self.transformer_en = Transformer_encoder_pad(d_model = 100)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)

        self.criterion = ContrastiveLoss_mask()
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, an_path, com_path, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        claploss = 0
        # anco_text_e, mask_text = self.bert_encoder(anco_text) 
        # anco_text_e = anco_text_e.cuda()
        # cross = calculate_cross(anco_text_e,anco_text_e)
        # anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp_an(anco_speech)
        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp_com(com_speech)

        if self.training:
            fea = aa_exatphone_pool(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label) 
            
            if fea != 1:
                an_audio_phnfea, phn_text, com_audio_phnfea, l1 = fea

            else:
                return 0

            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_speech_p2 = self.audio_adp_an2(an_audio_phnfea) 
            com_speech_p2 = self.audio_adp_com2(com_audio_phnfea)

            claploss = self.criterion(anco_speech_p2, com_speech_p2, phn_text)

        anco_speech_p = self.audio_adp_an2(anco_speech_p) 
        com_speech_p = self.audio_adp_com2(com_speech_p) 

        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(anco_speech.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4
    # 16,100,100

        # anco_text_p = anco_text_p.transpose(1, 2)
        # anco_speech_p = anco_speech_p.transpose(1, 2)
        # com_speech_p = com_speech_p.transpose(1, 2)


        # pattern_p = self.transformer(com_speech_p, com_speech_p, anco_speech_p, mask = None)
        # print(anchor_mask.shape, comparison_mask.transpose(1, 2).shape)
        # mask_cross = torch.bmm(mask_text.unsqueeze(-1).float(),comparison_mask.transpose(1, 2))
        # speech_s2_mask = torch.cat((anchor_mask, comparison_mask), dim=1)
        
        # mask_cross = torch.bmm(speech_s2_mask,speech_s2_mask.transpose(1, 2))

        # speech_text = torch.cat((anco_speech_p, com_speech_p), dim=1)
        # pattern_p = self.transformer_en(pad_anchor_speech, comparison_mask)
        pattern_p = self.transformer_en(pad_anchor_speech)


        # speech_text = torch.cat((anco_speech_p, anco_text_p), dim=1)
        # pattern_p = self.transformer(speech_text, mask = None)  #(B,407,128)

        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        out = self.gru(pattern_p)
        out = self.dense(out)
        return out, claploss



class udkws_aa2clap(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aa2clap, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)
        self.audio_adp_an2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp_com2 = Projection(input_dim=128, output_dim=128)
        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()
        # self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 3), stride=(2, 1), padding=(1, 1), output_padding=(8, 0))
        # self.upsample = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=True)
        # self.text_adp = text_CPEncoder(in_channels=768, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)
        # self.audio_adp = speech_CPEncoder(in_channels=384, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)

        # self.attn = nn.MultiheadAttention(embed_dim=embedding, num_heads=1)

        # self.transformer = Transformer_cross(embed_size = 128, num_layers = 1, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        # self.transformer_e = Transformer_self(embed_size = 128, num_layers = 4, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        self.transformer_en = Transformer_encoder_pad(d_model = 100)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)

        self.criterion = ContrastiveLoss_mask()
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, an_path, com_path, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        claploss = 0
        # anco_text_e, mask_text = self.bert_encoder(anco_text) 
        # anco_text_e = anco_text_e.cuda()
        # cross = calculate_cross(anco_text_e,anco_text_e)
        # anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp_an(anco_speech)
        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp_com(com_speech)

        if self.training:
            fea = aa_exatphone_pool(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label) 
            
            if fea != 1:
                an_audio_phnfea, phn_text, com_audio_phnfea, l1 = fea

            else:
                return 0

            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_speech_p2 = self.audio_adp_an2(an_audio_phnfea) 
            com_speech_p2 = self.audio_adp_com2(com_audio_phnfea)

            claploss = self.criterion(anco_speech_p2, com_speech_p2, phn_text)

        anco_speech_p = self.audio_adp_an2(anco_speech_p) 
        com_speech_p = self.audio_adp_com2(com_speech_p) 

        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(anco_speech.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4
    # 16,100,100

        # anco_text_p = anco_text_p.transpose(1, 2)
        # anco_speech_p = anco_speech_p.transpose(1, 2)
        # com_speech_p = com_speech_p.transpose(1, 2)


        # pattern_p = self.transformer(com_speech_p, com_speech_p, anco_speech_p, mask = None)
        # print(anchor_mask.shape, comparison_mask.transpose(1, 2).shape)
        # mask_cross = torch.bmm(mask_text.unsqueeze(-1).float(),comparison_mask.transpose(1, 2))
        # speech_s2_mask = torch.cat((anchor_mask, comparison_mask), dim=1)
        
        # mask_cross = torch.bmm(speech_s2_mask,speech_s2_mask.transpose(1, 2))

        # speech_text = torch.cat((anco_speech_p, com_speech_p), dim=1)
        # pattern_p = self.transformer_en(pad_anchor_speech, comparison_mask)
        pattern_p = self.transformer_en(pad_anchor_speech)


        # speech_text = torch.cat((anco_speech_p, anco_text_p), dim=1)
        # pattern_p = self.transformer(speech_text, mask = None)  #(B,407,128)

        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        out = self.gru(pattern_p)
        out = self.dense(out)
        return out, claploss



class udkws_aa2clapnoself(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aa2clapnoself, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)
        self.audio_adp_an2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp_com2 = Projection(input_dim=128, output_dim=128)
        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()
        # self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 3), stride=(2, 1), padding=(1, 1), output_padding=(8, 0))
        # self.upsample = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=True)
        # self.text_adp = text_CPEncoder(in_channels=768, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)
        # self.audio_adp = speech_CPEncoder(in_channels=384, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)

        # self.attn = nn.MultiheadAttention(embed_dim=embedding, num_heads=1)

        # self.transformer = Transformer_cross(embed_size = 128, num_layers = 1, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        # self.transformer_e = Transformer_self(embed_size = 128, num_layers = 4, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        # self.transformer_en = Transformer_encoder_pad(d_model = 100)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)

        self.criterion = ContrastiveLoss_mask()
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, an_path, com_path, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        claploss = 0
        # anco_text_e, mask_text = self.bert_encoder(anco_text) 
        # anco_text_e = anco_text_e.cuda()
        # cross = calculate_cross(anco_text_e,anco_text_e)
        # anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp_an(anco_speech)
        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp_com(com_speech)

        if self.training:
            fea = aa_exatphone_pool(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label) 
            
            if fea != 1:
                an_audio_phnfea, phn_text, com_audio_phnfea, l1 = fea

            else:
                return 0

            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_speech_p2 = self.audio_adp_an2(an_audio_phnfea) 
            com_speech_p2 = self.audio_adp_com2(com_audio_phnfea)

            claploss = self.criterion(anco_speech_p2, com_speech_p2, phn_text)

        anco_speech_p = self.audio_adp_an2(anco_speech_p) 
        com_speech_p = self.audio_adp_com2(com_speech_p) 


        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(anco_speech.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4
    # 16,100,100

        # anco_text_p = anco_text_p.transpose(1, 2)
        # anco_speech_p = anco_speech_p.transpose(1, 2)
        # com_speech_p = com_speech_p.transpose(1, 2)


        # pattern_p = self.transformer(com_speech_p, com_speech_p, anco_speech_p, mask = None)
        # print(anchor_mask.shape, comparison_mask.transpose(1, 2).shape)
        # mask_cross = torch.bmm(mask_text.unsqueeze(-1).float(),comparison_mask.transpose(1, 2))
        # speech_s2_mask = torch.cat((anchor_mask, comparison_mask), dim=1)
        
        # mask_cross = torch.bmm(speech_s2_mask,speech_s2_mask.transpose(1, 2))

        # speech_text = torch.cat((anco_speech_p, com_speech_p), dim=1)
        # pattern_p = self.transformer_en(pad_anchor_speech, comparison_mask)
        # pattern_p = self.transformer_en(pad_anchor_speech)
        pattern_p = pad_anchor_speech


        # speech_text = torch.cat((anco_speech_p, anco_text_p), dim=1)
        # pattern_p = self.transformer(speech_text, mask = None)  #(B,407,128)

        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        out = self.gru(pattern_p)
        out = self.dense(out)
        return out, claploss



class udkws_aclap(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aclap, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)
        self.audio_adp_an2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp_com2 = Projection(input_dim=128, output_dim=128)
        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()


        self.criterion = ContrastiveLoss_mask()
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, an_path, com_path, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        claploss = 0
        l1 = 0
        # anco_text_e, mask_text = self.bert_encoder(anco_text) 
        # anco_text_e = anco_text_e.cuda()
        # cross = calculate_cross(anco_text_e,anco_text_e)
        # anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp_an(anco_speech)
        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp_com(com_speech)

        if self.training:
            fea = aa_exatphone_pool(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label) 
            
            if fea != 1:
                an_audio_phnfea, phn_text, com_audio_phnfea, l1 = fea

            else:
                return 0

            anco_speech_p2 = self.audio_adp_an2(an_audio_phnfea) 
            com_speech_p2 = self.audio_adp_com2(com_audio_phnfea)

            claploss = self.criterion(anco_speech_p2, com_speech_p2, phn_text)

        anco_speech_p = self.audio_adp_an2(anco_speech_p) 
        com_speech_p = self.audio_adp_com2(com_speech_p) 


        return l1, claploss


class udkws_aclap_ma(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aclap_ma, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)
        self.audio_adp_an2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp_com2 = Projection(input_dim=128, output_dim=128)
        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()
        # self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 3), stride=(2, 1), padding=(1, 1), output_padding=(8, 0))
        # self.upsample = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=True)
        # self.text_adp = text_CPEncoder(in_channels=768, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)
        # self.audio_adp = speech_CPEncoder(in_channels=384, out_channels=128, kernel_size=3, stride=2, padding=1, pool_kernel_size=2, pool_stride=2)

        # self.attn = nn.MultiheadAttention(embed_dim=embedding, num_heads=1)

        # self.transformer = Transformer_cross(embed_size = 128, num_layers = 1, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        # self.transformer_e = Transformer_self(embed_size = 128, num_layers = 4, num_heads = 1, ff_hidden_size = 128, dropout = 0.1)
        # self.transformer_en = Transformer_encoder_pad(d_model = 100)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        # self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        # self.dense = nn.Linear(100, 1)
        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
                                    ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        self.criterion = ContrastiveLoss_mask()
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, an_path, com_path, label, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        claploss = 0
        l1 = 0
        # anco_text_e, mask_text = self.bert_encoder(anco_text) 
        # anco_text_e = anco_text_e.cuda()
        # cross = calculate_cross(anco_text_e,anco_text_e)
        # anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp_an(anco_speech)
        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp_com(com_speech)

        if self.training:
            # fea = aa_exatphone_pool_ma(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label, my_MA) 
            fea = aa_exatphone_pool(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label) 
            
            if fea != 1:
                an_audio_phnfea, phn_text, com_audio_phnfea, l1 = fea
            else:
                return 0

            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_speech_p2 = self.audio_adp_an2(an_audio_phnfea) 
            com_speech_p2 = self.audio_adp_com2(com_audio_phnfea)


            for k in range(len(phn_text)):
                new_phone_tmp = com_speech_p2[k].unsqueeze(0)
                old_phone = my_MA[self.idx2p[int(phn_text[k])]]
                thta = 0.8
                new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
                my_MA[self.idx2p[int(phn_text[k])]] = new_phone

            claploss = self.criterion(anco_speech_p2, com_speech_p2, phn_text)

        anco_speech_p = self.audio_adp_an2(anco_speech_p) 
        com_speech_p = self.audio_adp_com2(com_speech_p) 


        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        # cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)


        # out = self.gru(pattern_p)
        # out = self.dense(out)
        return l1, claploss, my_MA


class udkws_aa1clap(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aa1clap, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.transformer = Transformer_cross(embed_size = 100, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        # self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)

        self.criterion = ContrastiveLoss_mask()
        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
                                    ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, an_path, com_path, label, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        claploss = 0
        # anco_text_e, mask_text = self.bert_encoder(anco_text) 
        # anco_text_e = anco_text_e.cuda()
        # cross = calculate_cross(anco_text_e,anco_text_e)
        # anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp_an(anco_speech)
        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp_com(com_speech)

        if self.training:
            # fea = aa_exatphone_pool(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label) 
            fea = aa_exatphone_pool_2(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label) 
            
            if fea != 1:
                an_audio_phnfea, phn_text, com_audio_phnfea, l1 = fea

            else:
                return 0

            for k in range(len(phn_text)):
                new_phone_tmp = com_audio_phnfea[k].unsqueeze(0)
                old_phone = my_MA[self.idx2p[int(phn_text[k])]]
                thta = 0.8
                new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
                my_MA[self.idx2p[int(phn_text[k])]] = new_phone

            # anco_speech_p2 = self.audio_adp_an2(an_audio_phnfea) 
            # com_speech_p2 = self.audio_adp_com2(com_audio_phnfea)

            claploss = self.criterion(an_audio_phnfea, com_audio_phnfea, phn_text)

        # anco_speech_p = self.audio_adp_an2(anco_speech_p) 
        # com_speech_p = self.audio_adp_com2(com_speech_p) 

        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(anco_speech.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4
    # 16,100,100


        # pattern_p = self.transformer_en(pad_anchor_speech)

        max_corss_speech = pad_anchor_speech.max(dim = 1).values.unsqueeze(1)
        pattern_p = self.transformer(pad_anchor_speech, pad_anchor_speech, max_corss_speech)

        out = pattern_p.squeeze(1)

        # out = self.gru(pattern_p)
        out = self.dense(out)
        return out, claploss, my_MA

    
class udkws_aa1clapaug(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aa1clapaug, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.transformer = Transformer_cross(embed_size = 100, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        # self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)

        self.criterion = ContrastiveLoss_mask()
        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
                                    ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, an_path, com_path, label, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        claploss = 0

        anco_speech_p = self.audio_adp_an(anco_speech)

        com_speech_p = self.audio_adp_com(com_speech)

        if self.training:
            fea = aa_exatphone_pool(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label) 
            
            if fea != 1:
                an_audio_phnfea, phn_text, com_audio_phnfea, l1 = fea

            else:
                return 0

            # for k in range(len(phn_text)):
            #     new_phone_tmp = com_audio_phnfea[k].unsqueeze(0)
            #     old_phone = my_MA[self.idx2p[int(phn_text[k])]]
            #     thta = 0.8
            #     new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
            #     my_MA[self.idx2p[int(phn_text[k])]] = new_phone

            # anco_speech_p2 = self.audio_adp_an2(an_audio_phnfea) 
            # com_speech_p2 = self.audio_adp_com2(com_audio_phnfea)

            claploss = self.criterion(an_audio_phnfea, com_audio_phnfea, phn_text)

            ###################   aug   #######################
            # com_speech_p, comparison_mask, anco_speech_p, anchor_mask, label = augphone_ma(com_speech_p, comparison_mask, com_path, anco_speech_p, anchor_mask, label, my_MA) 

            anco_speech_p, anchor_mask, com_speech_p, comparison_mask, label = augphone_ma(anco_speech_p, anchor_mask, com_path, com_speech_p, comparison_mask, label, my_MA) 


        # anco_speech_p = self.audio_adp_an2(anco_speech_p) 
        # com_speech_p = self.audio_adp_com2(com_speech_p) 


        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(com_speech_p.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4
    # 16,100,100


        # pattern_p = self.transformer_en(pad_anchor_speech)

        max_corss_speech = pad_anchor_speech.max(dim = 1).values.unsqueeze(1)
        pattern_p = self.transformer(pad_anchor_speech, pad_anchor_speech, max_corss_speech)

        out = pattern_p.squeeze(1)

        # out = self.gru(pattern_p)
        out = self.dense(out)
        return out, claploss, my_MA, label


    
class udkws_aa1clapaug3(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aa1clapaug3, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.transformer = Transformer_cross(embed_size = 100, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        # self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)

        self.criterion = ContrastiveLoss_mask()
        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
                                    ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, an_path, com_path, label, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        claploss = 0

        anco_speech_p = self.audio_adp_an(anco_speech)

        com_speech_p = self.audio_adp_com(com_speech)

        if self.training:
            fea = aa_exatphone_pool(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label) 
            
            if fea != 1:
                an_audio_phnfea, phn_text, com_audio_phnfea, l1 = fea

            else:
                return 0

            # for k in range(len(phn_text)):
            #     new_phone_tmp = com_audio_phnfea[k].unsqueeze(0)
            #     old_phone = my_MA[self.idx2p[int(phn_text[k])]]
            #     thta = 0.8
            #     new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
            #     my_MA[self.idx2p[int(phn_text[k])]] = new_phone

            # anco_speech_p2 = self.audio_adp_an2(an_audio_phnfea) 
            # com_speech_p2 = self.audio_adp_com2(com_audio_phnfea)

            claploss = self.criterion(an_audio_phnfea, com_audio_phnfea, phn_text)

            ###################   aug   #######################
            # com_speech_p, comparison_mask, anco_speech_p, anchor_mask, label = augphone_ma(com_speech_p, comparison_mask, com_path, anco_speech_p, anchor_mask, label, my_MA) 

            anco_speech_p, anchor_mask, com_speech_p, comparison_mask, label = augphone_ma3hard(anco_speech_p, anchor_mask, com_path, com_speech_p, comparison_mask, label, my_MA) 


        # anco_speech_p = self.audio_adp_an2(anco_speech_p) 
        # com_speech_p = self.audio_adp_com2(com_speech_p) 


        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(com_speech_p.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4
    # 16,100,100


        # pattern_p = self.transformer_en(pad_anchor_speech)

        max_corss_speech = pad_anchor_speech.max(dim = 1).values.unsqueeze(1)
        pattern_p = self.transformer(pad_anchor_speech, pad_anchor_speech, max_corss_speech)

        out = pattern_p.squeeze(1)

        # out = self.gru(pattern_p)
        out = self.dense(out)
        return out, claploss, my_MA, label



class udkws_aa1clap1proc(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aa1clap1proc, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        # self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.transformer = Transformer_cross(embed_size = 100, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        # self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)

        self.criterion = ContrastiveLoss_mask()
        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
                                    ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, an_path, com_path, label, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        claploss = 0
        # anco_text_e, mask_text = self.bert_encoder(anco_text) 
        # anco_text_e = anco_text_e.cuda()
        # cross = calculate_cross(anco_text_e,anco_text_e)
        # anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp_com(anco_speech)
        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp_com(com_speech)

        if self.training:
            fea = aa_exatphone_pool(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label) 
            
            if fea != 1:
                an_audio_phnfea, phn_text, com_audio_phnfea, l1 = fea

            else:
                return 0

            for k in range(len(phn_text)):
                new_phone_tmp = com_audio_phnfea[k].unsqueeze(0)
                old_phone = my_MA[self.idx2p[int(phn_text[k])]]
                thta = 0.8
                new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
                my_MA[self.idx2p[int(phn_text[k])]] = new_phone

            # anco_speech_p2 = self.audio_adp_an2(an_audio_phnfea) 
            # com_speech_p2 = self.audio_adp_com2(com_audio_phnfea)

            claploss = self.criterion(an_audio_phnfea, com_audio_phnfea, phn_text)

        # anco_speech_p = self.audio_adp_an2(anco_speech_p) 
        # com_speech_p = self.audio_adp_com2(com_speech_p) 

        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(anco_speech.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4
    # 16,100,100


        # pattern_p = self.transformer_en(pad_anchor_speech)

        max_corss_speech = pad_anchor_speech.max(dim = 1).values.unsqueeze(1)
        pattern_p = self.transformer(pad_anchor_speech, pad_anchor_speech, max_corss_speech)

        out = pattern_p.squeeze(1)

        # out = self.gru(pattern_p)
        out = self.dense(out)
        return out, claploss, my_MA


class udkws_aa1clapatt(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aa1clapatt, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.transformer = Transformer_cross(embed_size = 100, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        # self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)

        self.transformer_en = Transformer_encoder_pad(d_model = 100)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.denseatt = nn.Linear(100, 1)

        self.criterion = ContrastiveLoss_mask()
        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
                                    ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
        
        # self.transformer_en = Transformer_encoder_pad(d_model = 100)
        # # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        # self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        # self.dense = nn.Linear(100, 1)
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, an_path, com_path, label, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        claploss = 0
        # anco_text_e, mask_text = self.bert_encoder(anco_text) 
        # anco_text_e = anco_text_e.cuda()
        # cross = calculate_cross(anco_text_e,anco_text_e)
        # anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp_an(anco_speech)
        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp_com(com_speech)

        if self.training:
            fea = aa_exatphone_pool(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label) 
            
            if fea != 1:
                an_audio_phnfea, phn_text, com_audio_phnfea, l1 = fea

            else:
                return 0

            for k in range(len(phn_text)):
                new_phone_tmp = com_audio_phnfea[k].unsqueeze(0)
                old_phone = my_MA[self.idx2p[int(phn_text[k])]]
                thta = 0.8
                new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
                my_MA[self.idx2p[int(phn_text[k])]] = new_phone

            # anco_speech_p2 = self.audio_adp_an2(an_audio_phnfea) 
            # com_speech_p2 = self.audio_adp_com2(com_audio_phnfea)

            claploss = self.criterion(an_audio_phnfea, com_audio_phnfea, phn_text)

        # anco_speech_p = self.audio_adp_an2(anco_speech_p) 
        # com_speech_p = self.audio_adp_com2(com_speech_p) 

        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(anco_speech.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4
    # 16,100,100


        # pattern_p = self.transformer_en(pad_anchor_speech)

        max_corss_speech = pad_anchor_speech.max(dim = 1).values.unsqueeze(1)
        pattern_p = self.transformer(pad_anchor_speech, pad_anchor_speech, max_corss_speech)

        out = pattern_p.squeeze(1)

        # out = self.gru(pattern_p)
        out = self.dense(out)

        pattern_patt = self.transformer_en(pad_anchor_speech)
        outatt = self.gru(pattern_patt)
        outatt = self.denseatt(outatt)
        out = out + outatt

        return out, claploss, my_MA




class udkws_aa1clap3class(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_aa1clap3class, self).__init__()
        # self.bert_encoder = BERT_encoder() 
        # self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp_an = Projection(input_dim=384, output_dim=128)
        self.audio_adp_com = Projection(input_dim=384, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.transformer = Transformer_cross(embed_size = 100, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)

        # self.gru = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        # self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)

        self.transformer_3cls = Transformer_cross(embed_size = 100, num_layers = 1, num_heads = 1, ff_hidden_size = 100, dropout = 0.1)

        self.dense_3cls = nn.Linear(100, 3)

        self.criterion = ContrastiveLoss_mask()
        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
                                    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH',
                                    'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
                                    'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
                                    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0',
                                    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1',
                                    'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH',
                                    ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
    
    def forward(self, anco_speech, anchor_mask, com_speech, comparison_mask, anco_text, an_path, com_path, label, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 128, 768]
        '''
        unsample_list = []
        claploss = 0
        out3 = 0
        # anco_text_e, mask_text = self.bert_encoder(anco_text) 
        # anco_text_e = anco_text_e.cuda()
        # cross = calculate_cross(anco_text_e,anco_text_e)
        # anco_text_p = self.text_adp(anco_text_e)

        # anco_speech = anco_speech.transpose(1, 2)
        anco_speech_p = self.audio_adp_an(anco_speech)
        # cosine_sim_matrix2 = calculate_cross(anco_speech, com_speech, anchor_mask, comparison_mask)
        # com_speech = com_speech.transpose(1, 2)
        com_speech_p = self.audio_adp_com(com_speech)

        if self.training:
            fea = aa_exatphone_pool(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label) 
            # fea = aa_exatphone_pool_2(anco_speech_p, anchor_mask, an_path, com_speech_p, comparison_mask, com_path, label) 
            
            if fea != 1:
                an_audio_phnfea, phn_text, com_audio_phnfea, l1 = fea

            else:
                return 0

            for k in range(len(phn_text)):
                new_phone_tmp = com_audio_phnfea[k].unsqueeze(0)
                old_phone = my_MA[self.idx2p[int(phn_text[k])]]
                thta = 0.8
                new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
                my_MA[self.idx2p[int(phn_text[k])]] = new_phone
        
            # anco_speech_p2 = self.audio_adp_an2(an_audio_phnfea) 
            # com_speech_p2 = self.audio_adp_com2(com_audio_phnfea)

            claploss = self.criterion(an_audio_phnfea, com_audio_phnfea, phn_text)

        # anco_speech_p = self.audio_adp_an2(anco_speech_p) 
        # com_speech_p = self.audio_adp_com2(com_speech_p) 

        cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p, anchor_mask, comparison_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(anco_speech.shape[0]):
            len_anco = anchor_mask[i].sum()
            len_com = comparison_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(100, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4 
    # 16,100,100 


        # pattern_p = self.transformer_en(pad_anchor_speech)

        max_corss_speech = pad_anchor_speech.max(dim = 1).values.unsqueeze(1)
        pattern_p = self.transformer(pad_anchor_speech, pad_anchor_speech, max_corss_speech)

        out = pattern_p.squeeze(1)
        if self.training:

            pattern_p3 = self.transformer_3cls(pad_anchor_speech, pad_anchor_speech, max_corss_speech)
            out3 = pattern_p3.squeeze(1)
            out3 = self.dense_3cls(out3)


        # out = self.gru(pattern_p)
        out = self.dense(out)
        return out, out3, claploss, my_MA



# text = ['is a city','sdfs']
# # text = ['is a city']


# # bert = BERT_encoder() 
# # output_text = bert(text) 
# # print(output_text.shape)


# input_embedding = torch.randn(2, 1500, 384)
# conv_pool_encoder = udkws()

# output_embedding = conv_pool_encoder(input_embedding,text,input_embedding)  # (8, 128, T)

# print(output_embedding.shape)  #  (8, T, 128)

