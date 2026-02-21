import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append("/train20/intern/permanent/kwli2/udkws/attakws")
from transformer2 import Transformer_self, Transformer_cross, Transformer_encoder, CrossAttentionLayer, Transformer_encoder_pad
from model.cosafind import *
from torch.nn.utils.rnn import pad_sequence
from transformers import DebertaV2Tokenizer, AutoProcessor, WhisperProcessor
from model.clap_phn import ContrastiveLoss_mask, ContrastiveLossword_mask
from model.fadata import *



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


class BERT_encoder(nn.Module):
    def __init__(self,):
        super(BERT_encoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('/train20/intern/permanent/kwli2/udkws/tmp1/bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('/train20/intern/permanent/kwli2/udkws/tmp1/bert-base-uncased')
        self.bert_model = self.bert_model.cuda()
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(self, antext, comtext, path):
           

        for key in inputs:
            inputs[key] = inputs[key].cuda()
        # inputs = inputs.cuda()
        mask = inputs['attention_mask']
        outputs = self.bert_model(**inputs)
        x = outputs.last_hidden_state
        return x, mask

class BERT_encoder_word(nn.Module):
    def __init__(self,):
        super(BERT_encoder_word, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('/train20/intern/permanent/kwli2/udkws/tmp1/bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('/train20/intern/permanent/kwli2/udkws/tmp1/bert-base-uncased')
        self.bert_model = self.bert_model.cuda()
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(self, textan, textcom):
        # print('an: ', textan)
        # print('com: ',textcom)
        inputsan = self.tokenizer(textan, return_tensors='pt', max_length=10, padding='max_length', truncation=True)
        inputscom = self.tokenizer(textcom, return_tensors='pt', max_length=10, padding='max_length', truncation=True)
        for key in inputsan:
            inputsan[key] = inputsan[key].cuda()
        for key in inputscom:
            inputscom[key] = inputscom[key].cuda()
        # tokenan = inputsan.input_ids
        # tokencom = inputscom.input_ids
        maskan_cls = (inputsan['input_ids']!=101)&(inputsan['input_ids']!=102)

        an_id = inputsan['input_ids'][(inputsan['input_ids']!=101)&(inputsan['input_ids']!=102)].view(-1,8)
        com_id = inputscom['input_ids'][(inputscom['input_ids']!=101)&(inputscom['input_ids']!=102)].view(-1,8)
        label_phone = an_id == com_id 
        label_phone = label_phone.float()
        # inputsan['attention_mask'] = inputsan['attention_mask'][:, 2:]
        mask = inputsan['attention_mask'][:, 2:]
        outputs = self.bert_model(input_ids = inputsan['input_ids'], attention_mask = inputsan['attention_mask'])
        x = outputs.last_hidden_state
        x = x[maskan_cls].view(-1, 8, 768)
        return x, mask, label_phone


class BERT_encoder_wordanS(nn.Module):
    def __init__(self,):
        super(BERT_encoder_wordanS, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('/train20/intern/permanent/kwli2/udkws/tmp1/bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('/train20/intern/permanent/kwli2/udkws/tmp1/bert-base-uncased')
        self.bert_model = self.bert_model.cuda()
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(self, textan, textcom):
        # print('an: ', textan)
        # print('com: ',textcom)
        inputsan = self.tokenizer(textan, return_tensors='pt', max_length=10, padding='max_length', truncation=True)
        inputscom = self.tokenizer(textcom, return_tensors='pt', max_length=10, padding='max_length', truncation=True)
        for key in inputsan:
            inputsan[key] = inputsan[key].cuda()
        for key in inputscom:
            inputscom[key] = inputscom[key].cuda()
        # tokenan = inputsan.input_ids
        # tokencom = inputscom.input_ids
        # maskan_cls = (inputsan['input_ids']!=101)&(inputsan['input_ids']!=102)
        maskan_cls = (inputsan['input_ids']!=102)


        an_id = inputsan['input_ids'][(inputsan['input_ids']!=101)&(inputsan['input_ids']!=102)].view(-1,8)
        com_id = inputscom['input_ids'][(inputscom['input_ids']!=101)&(inputscom['input_ids']!=102)].view(-1,8)
        label_phone = an_id == com_id 
        label_phone = label_phone.float()
        # inputsan['attention_mask'] = inputsan['attention_mask'][:, 2:]
        mask = inputsan['attention_mask'][:, 1:]
        outputs = self.bert_model(input_ids = inputsan['input_ids'], attention_mask = inputsan['attention_mask'])
        x = outputs.last_hidden_state
        x = x[maskan_cls].view(-1, 9, 768)
        return x, mask, label_phone



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


class GRUFCModel_mask(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUFCModel_mask, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask = None):
        # x = x * mask   # gru_out[0,:,0]
        gru_out, _ = self.gru(x)
        # gru_out = gru_out * mask
        B, S, _ = x.shape
        lenmask = mask[:,100:,0].sum(dim = 1) + 99
        lenmask = lenmask.to(torch.long)
        mask_t = torch.zeros((B, S), dtype=torch.bool).cuda()
        mask_t[torch.arange(B, dtype=torch.long), lenmask] = True
        mask_t = mask_t.unsqueeze(-1).expand(-1, gru_out.shape[1], gru_out.shape[2])
        # gru_last_output = gru_out[:, -1, :]
        gru_last_output = gru_out[mask_t].view(-1,128)
        fc_out = self.fc(gru_last_output)
        return fc_out

class GRUFCModel_maskB(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUFCModel_maskB, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask = None):
        # x = x * mask   # gru_out[0,:,0]
        gru_out, _ = self.gru(x)
        # gru_out = gru_out * mask
        B, S, _ = x.shape
        lenmask = mask[:,140:,0].sum(dim = 1) + 139
        lenmask = lenmask.to(torch.long)
        mask_t = torch.zeros((B, S), dtype=torch.bool).cuda()
        mask_t[torch.arange(B, dtype=torch.long), lenmask] = True
        mask_t = mask_t.unsqueeze(-1).expand(-1, gru_out.shape[1], gru_out.shape[2])
        # gru_last_output = gru_out[:, -1, :]
        gru_last_output = gru_out[mask_t].view(-1,128)
        fc_out = self.fc(gru_last_output)
        return fc_out


class GRUFCModel_maskBcat(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUFCModel_maskBcat, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask = None):
        # x = x * mask   # gru_out[0,:,0]
        B, t, c = x.shape
        x1 = []
        x1_mask = []

        for i in range(B):
            x_tmp = torch.masked_select(x[i],mask[i].bool()).view(-1,c)
            x1.append(x_tmp)
            mask_tmp = torch.masked_select(mask[i].float(),mask[i].bool()).unsqueeze(-1)
            x1_mask.append(mask_tmp)
        x1 = pad_sequence(x1, batch_first=True, padding_value=0.0)
        x1_mask = pad_sequence(x1_mask, batch_first=True, padding_value=0.0)

        gru_out, _ = self.gru(x1)
        # # gru_out = gru_out * mask
        B, S, _ = x1.shape
        lenmask = x1_mask.sum(dim = 1) -1
        lenmask = lenmask.to(torch.long)
        # # lenmask = mask[:,140:,0].sum(dim = 1) + 139
        # # lenmask = lenmask.to(torch.long)
        mask_t = torch.zeros((B, S), dtype=torch.bool).cuda()
        mask_t[torch.arange(B, dtype=torch.long), lenmask.squeeze(-1)] = True
        mask_t = mask_t.unsqueeze(-1).expand(-1, gru_out.shape[1], gru_out.shape[2])
        # gru_last_output = gru_out[:, -1, :]

        gru_last_output = gru_out[mask_t].view(-1,128)

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



class CLAP_encoder(nn.Module):
    def __init__(self,):
        super(CLAP_encoder, self).__init__()
        self.tokenizer = DebertaV2Tokenizer.from_pretrained('/train20/intern/permanent/kwli2/udkws/clap-ipa-main/models--charsiu--IPATokenizer')
        self.phone_encoder = PhoneEncoder.from_pretrained('/train20/intern/permanent/kwli2/udkws/clap-ipa-main/models--anyspeech--clap-ipa-tiny-phone')
        self.len = 40
        # self.tokenizer = BertTokenizer.from_pretrained('/train20/intern/permanent/kwli2/udkws/tmp1/bert-base-uncased')
        # self.bert_model = BertModel.from_pretrained('/train20/intern/permanent/kwli2/udkws/tmp1/bert-base-uncased')
        # self.bert_model = self.bert_model.cuda()
        for param in self.phone_encoder.parameters():
            param.requires_grad = False

    def forward(self, text):
        phonelist = []
        phonemasklist = []
        for t in text:
            ipa_input = self.tokenizer(t, return_tensors="pt").input_ids
            ipa_input = ipa_input.cuda()
            phone_embed = self.phone_encoder(ipa_input).last_hidden_state
            mask_tmp = torch.ones(phone_embed.shape[1]).cuda()
            padding_len = self.len - phone_embed.shape[1]
            padphone_embed = F.pad(phone_embed, (0, 0, 0, padding_len))
            padmask_tmp = F.pad(mask_tmp, (0, padding_len))
            phonelist.append(padphone_embed)
            phonemasklist.append(padmask_tmp)
        phone = pad_sequence(phonelist, batch_first=True, padding_value=0.0).squeeze(1)  #16,40,384
        mask = pad_sequence(phonemasklist, batch_first=True, padding_value=0.0)  #10080,4

        return phone, mask



class SequenceCrossEntropy(nn.Module):
    def forward(self, speech_label, text_label, logits, mask, reduction='sum'):
        # Data pre-processing
        if text_label.size(1) > speech_label.size(1):
            speech_label = F.pad(speech_label, (0, text_label.size(1) - speech_label.size(1)), 'constant', 0)
        elif text_label.size(1) < speech_label.size(1):
            speech_label = speech_label[:, :text_label.size(1)]

        # Make paired data between text and speech phonemes
        # paired_label = (text_label == speech_label) & logits._keras_mask
        paired_label = (text_label == speech_label)
        # paired_label = paired_label.float().view(-1, 1)
        # logits = logits[:,-30:,0]
        paired_label = torch.masked_select(paired_label, mask.bool()).unsqueeze(1)
        logits = torch.masked_select(logits, mask.bool()).unsqueeze(1)
        # lenlog = text_label.shape[-1]
        # logits = logits[:,-lenlog:,0]
        # logits = logits.view(-1, 1)
        # logits = logits.reshape(-1,1)

        # Get BinaryCrossEntropy loss
        # bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        bce_loss = nn.BCEWithLogitsLoss()

        # loss = bce_loss(paired_label, logits)#8384,1920
        loss = bce_loss(logits.float(), paired_label.float())#8384,1920
        if reduction == 'sum':
            loss = loss / logits.size(0)  # divide by batch size
            loss = loss * speech_label.size(0)  # multiply by batch size

        return loss 



class udkws_atcat_wordSen(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_wordSen, self).__init__()
        self.bert_encoder = BERT_encoder_wordanS() 
        self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel(input_dim=128, hidden_dim=128, output_dim=128)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
        self.dense_phone = nn.Linear(128, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()

    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''
        anco_text_e, mask_text, phone_label = self.bert_encoder(anco_text, com_text) 
        anco_text_e = anco_text_e.cuda()
        mask_text = mask_text.unsqueeze(-1).float()
        if len(label.shape) == 1:
            phone_label = torch.cat((label.unsqueeze(-1), phone_label), dim=1)
        else:
            phone_label = torch.cat((label, phone_label), dim=1)

        # phone_label = torch.cat((label.unsqueeze(-1), phone_label), dim=1)

        anco_text_p = self.text_adp(anco_text_e) 

        anco_speech_p = self.audio_adp(anco_speech)

        com_speech_p = self.audio_adp(com_speech)

        speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        phone_brach = pattern_p[:,100:,:] #b, 8,128
        
        out_phone = self.dense_phone(phone_brach)

        paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        loss = self.bce_loss(logits.float(), paired_label.float()) #8384,1920
        
        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        out = self.gru(pattern_p)
        out = self.dense(out)

        return out, loss


class udkws_atcat_word(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_word, self).__init__()
        self.bert_encoder = BERT_encoder_word() 
        self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel(input_dim=128, hidden_dim=128, output_dim=128)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
        self.dense_phone = nn.Linear(128, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()

    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''
        anco_text_e, mask_text, phone_label = self.bert_encoder(anco_text, com_text) 
        anco_text_e = anco_text_e.cuda()
        mask_text = mask_text.unsqueeze(-1).float()

        anco_text_p = self.text_adp(anco_text_e) 
        anco_speech_p = self.audio_adp(anco_speech)
        com_speech_p = self.audio_adp(com_speech)

        speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        phone_brach = pattern_p[:,100:,:] #b, 8,128
        
        out_phone = self.dense_phone(phone_brach)

        paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        
        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        out = self.gru(pattern_p)
        out = self.dense(out)

        return out, loss

# best
class udkws_atcat_phone(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phone, self).__init__()
        # self.bert_encoder = BERT_encoder_word() 
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.transformer_en = Transformer_encoder(nlayers = 4)

        self.gru = GRUFCModel_mask(input_dim=128, hidden_dim=128, output_dim=128)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
        self.dense_phone = nn.Linear(128, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()

    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''
        # anco_text_e, mask_text, phone_label = self.bert_encoder(anco_text, com_text) 
        phone_label = ang2p == comg2p
        # anco_text_e = anco_text_e.cuda()
        # mask_text = mask_text.unsqueeze(-1).float()

        # lengths = torch.tensor([len(sublist) for sublist in anphn])
        # mask_text = torch.zeros((len(anphn), 40), dtype=torch.float)

        # row_indices = torch.arange(mask_text.size(1)).expand(len(anphn), -1)
        # lengths_expanded = lengths.unsqueeze(1).expand_as(mask_text)
        # mask_text = (row_indices < lengths_expanded).float()
        # mask_text = mask_text.unsqueeze(-1)
        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 

        anco_speech_p = self.audio_adp(anco_speech)

        com_speech_p = self.audio_adp(com_speech)

        speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        phone_brach = pattern_p[:,100:,:] #b, 8,128
        
        out_phone = self.dense_phone(phone_brach)

        paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        

        out = self.gru(pattern_p, speech_text_mask)
        out = self.dense(out)

        return out, loss


class udkws_atcat_faphone(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_faphone, self).__init__()
        # self.bert_encoder = BERT_encoder_word() 
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel_mask(input_dim=128, hidden_dim=128, output_dim=128)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
        self.dense_phone = nn.Linear(128, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''
        # anco_text_e, mask_text, phone_label = self.bert_encoder(anco_text, com_text) 
        phone_label = ang2p == comg2p
        # anco_text_e = anco_text_e.cuda()
        # mask_text = mask_text.unsqueeze(-1).float()

        # lengths = torch.tensor([len(sublist) for sublist in anphn])
        # mask_text = torch.zeros((len(anphn), 40), dtype=torch.float)

        # row_indices = torch.arange(mask_text.size(1)).expand(len(anphn), -1)
        # lengths_expanded = lengths.unsqueeze(1).expand_as(mask_text)
        # mask_text = (row_indices < lengths_expanded).float()
        # mask_text = mask_text.unsqueeze(-1)
        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 

        anco_speech_p = self.audio_adp(anco_speech)

        com_speech_p = self.audio_adp(com_speech)

        speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        out_phone = self.dense_phone(phone_brach) 

        paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        out = self.gru(pattern_p, speech_text_mask)
        out = self.dense(out)

        return out, loss


class udkws_atcat_clapphone(nn.Module):  # only clap
    def __init__(self, embedding = 128):
        super(udkws_atcat_clapphone, self).__init__()
        # self.bert_encoder = BERT_encoder_word() 
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.criterion = ContrastiveLoss_mask()

        # self.transformer_en = Transformer_encoder()

        # self.gru = GRUFCModel_mask(input_dim=128, hidden_dim=128, output_dim=128)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        # self.dense = nn.Linear(128, 1)
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''

        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)

        com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)

        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        fea = exatphone_pool_pip(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
        fea2 = exatphone_pool_pip(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
        
        if fea != 1:
            audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
            l_all = l1
            audio_phnfea = audio_phnfea1
            phn_text_embd2 = phn_text_embd21
            phn_text = phn_text1
        if fea2 !=1:
            audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
            l_all = l2
            audio_phnfea = audio_phnfea2
            phn_text_embd2 = phn_text_embd22
            phn_text = phn_text2


        if fea2 !=1 and fea != 1:
            audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
            phn_text = torch.cat((phn_text1, phn_text2), dim=0)
            phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
            l_all = l1 + l2

        if fea2 == 1 and fea == 1:
            return 0
        # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
        # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

        anco_text_p2 = self.text_adp2(phn_text_embd2) 
        anco_speech_p2 = self.audio_adp2(audio_phnfea.float())


        claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)

        out = l_all

        # speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        # mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        # speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        # pattern_p = self.transformer_en(speech_text, mask_cross) 
        # pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        # out_phone = self.dense_phone(phone_brach) 

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        # out = self.gru(pattern_p, speech_text_mask)
        # out = self.dense(out)

        return out, claploss

class udkws_atcat_phonecat(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecat, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel_mask(input_dim=128, hidden_dim=128, output_dim=128)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]
        '''

        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)

        com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        

        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)

        # out = l_all

        speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        # out_phone = self.dense_phone(phone_brach) 

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        out = self.gru(pattern_p, speech_text_mask)
        out = self.dense(out)

        return out, claploss


class udkws_atcat_phonecat2(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecat2, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel_mask(input_dim=128, hidden_dim=128, output_dim=128)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]
        '''

        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)

        com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        

        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            else:
                return 0
            # if fea2 !=1:
            #     audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
            #     l_all = l2
            #     audio_phnfea = audio_phnfea2
            #     phn_text_embd2 = phn_text_embd22
            #     phn_text = phn_text2


            # if fea2 !=1 and fea != 1:
            #     audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
            #     phn_text = torch.cat((phn_text1, phn_text2), dim=0)
            #     phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
            #     l_all = l1 + l2

            # if fea2 == 1 and fea == 1:
            #     return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)


        speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        # out_phone = self.dense_phone(phone_brach) 

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        out = self.gru(pattern_p, speech_text_mask)
        out = self.dense(out)

        return out, claploss


class udkws_atcat_phonecatfc4(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecatfc4, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel_mask(input_dim=128, hidden_dim=128, output_dim=128)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]
        '''

        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)

        com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
    
        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0

            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)

        anco_text_p = self.text_adp2(anco_text_p) 
        com_speech_p = self.audio_adp2(com_speech_p.float()) 

        speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        # out_phone = self.dense_phone(phone_brach) 

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        out = self.gru(pattern_p, speech_text_mask)
        out = self.dense(out)

        return out, claploss



class udkws_atcat_phonecatpip(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecatpip, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel_mask(input_dim=128, hidden_dim=128, output_dim=128)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]
        '''

        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 

        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        

        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            anco_speech_p = self.audio_adp(anco_speech)
            com_text_p = self.text_adp(comphn.float()) 
            fea = exatphone_pool_pip(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool_pip(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)

        # out = l_all

        speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        # out_phone = self.dense_phone(phone_brach) 

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        out = self.gru(pattern_p, speech_text_mask)
        out = self.dense(out)

        return out, claploss


class udkws_atcat_phonecatpipfc4(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecatpipfc4, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel_mask(input_dim=128, hidden_dim=128, output_dim=128)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]
        '''

        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)

        com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        

        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool_pip(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool_pip(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)

        # out = l_all

        anco_text_p = self.text_adp2(anco_text_p) 
        com_speech_p = self.audio_adp2(com_speech_p.float())

        speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        # out_phone = self.dense_phone(phone_brach) 

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        out = self.gru(pattern_p, speech_text_mask)
        out = self.dense(out)

        return out, claploss
# torch.masked_select(phn_text_embd, phn_mask.bool().unsqueeze(-1)).unsqueeze(1).view(-1,128)


class udkws_atcat_phonecatfc4phn(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecatfc4phn, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel_mask(input_dim=128, hidden_dim=128, output_dim=128)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
        self.dense_phone = nn.Linear(128, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path):
    # def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]
        '''
        phone_label = ang2p == comg2p
        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)

        com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
    
        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0

            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)

        anco_text_p = self.text_adp2(anco_text_p) 
        com_speech_p = self.audio_adp2(com_speech_p.float()) 

        speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        out_phone = self.dense_phone(phone_brach) 

        paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        lossce = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        out = self.gru(pattern_p, speech_text_mask)
        out = self.dense(out)

        return out, claploss, lossce



class udkws_wordphn(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_wordphn, self).__init__()
        self.bert_encoder = BERT_encoder_word() 
        self.textbert_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        # self.textbert_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.textg2p_adp = Projection(input_dim=256, output_dim=128)
        self.textg2p_adp2 = Projection(input_dim=128, output_dim=128)

        self.criterion = ContrastiveLoss_mask()

        self.bertcriterion = ContrastiveLossword_mask()



        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel_maskBcat(input_dim=128, hidden_dim=128, output_dim=128)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()

    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''
        anco_text_e, mask_bert_textan, phone_label = self.bert_encoder(anco_text, com_text) 
        anco_text_e = anco_text_e.cuda()
        mask_textan = mask_textan.unsqueeze(-1).float()

        anco_text_b = self.textbert_adp(anco_text_e) 


        anco_speech_p = self.audio_adp(anco_speech)
        com_speech_p = self.audio_adp(com_speech)

        anco_text_p = self.textg2p_adp(anphn.float()) 
        com_text_p = self.textg2p_adp(comphn.float()) 
        bertclaploss = 0
        claploss = 0


        if self.training:
            ## BERT  ###
            bertfea = exatword_pool(anco_speech_p, anco_mask, an_path, anco_text, anco_text_b, mask_bert_textan) 
            audio_bertfea1, bert_text1, bert_text_embd21, l1 = bertfea

            bertclaploss = self.bertcriterion(audio_bertfea1, bert_text_embd21, bert_text1)

            
            ## G2P  ###
            fea = exatphone_pool_pip(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool_pip(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0

            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.textg2p_adp2(phn_text_embd2) 
            anco_speech_p2 = self.textg2p_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)


            

        speech_text_mask = torch.cat((com_mask, mask_textan.squeeze(-1), mask_bert_textan.unsqueeze(-1)), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p, anco_text_b), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128
        
        # out_phone = self.dense_phone(phone_brach)

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920


        out = self.gru(pattern_p, speech_text_mask)
        out = self.dense(out)

        return out, bertclaploss, claploss


class udkws_wordphnS(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_wordphnS, self).__init__()
        self.bert_encoder = BERT_encoder_word() 
        self.textbert_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        # self.textbert_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.textg2p_adp = Projection(input_dim=256, output_dim=128)
        self.textg2p_adp2 = Projection(input_dim=128, output_dim=128)

        self.criterion = ContrastiveLoss_mask()

        self.bertcriterion = ContrastiveLossword_mask()



        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel_maskBcat(input_dim=128, hidden_dim=128, output_dim=128)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()

    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''
        anco_text_e, mask_bert_textan, phone_label = self.bert_encoder(anco_text, com_text) 
        anco_text_e = anco_text_e.cuda()
        mask_textan = mask_textan.unsqueeze(-1).float()

        anco_text_b = self.textbert_adp(anco_text_e) 


        anco_speech_p = self.audio_adp(anco_speech)
        com_speech_p = self.audio_adp(com_speech)

        anco_text_p = self.textg2p_adp(anphn.float()) 
        com_text_p = self.textg2p_adp(comphn.float()) 
        bertclaploss = 0
        claploss = 0


        if self.training:
            ## BERT  ###
            bertfea = exatword_pool(anco_speech_p, anco_mask, an_path, anco_text, anco_text_b, mask_bert_textan) 
            audio_bertfea1, bert_text1, bert_text_embd21, l1 = bertfea

            bertclaploss = self.bertcriterion(audio_bertfea1, bert_text_embd21, bert_text1)

            
            ## G2P  ###
            fea = exatphone_pool_pip(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool_pip(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0

            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.textg2p_adp2(phn_text_embd2) 
            anco_speech_p2 = self.textg2p_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)


            

        speech_text_mask = torch.cat((com_mask, mask_textan.squeeze(-1), mask_bert_textan.unsqueeze(-1)), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p, anco_text_b), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128
        
        # out_phone = self.dense_phone(phone_brach)

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920


        out = self.gru(pattern_p, speech_text_mask)
        out = self.dense(out)

        return out, bertclaploss, claploss



def gaussian_kernel(size, sigma):
        coords = torch.tensor([x - (size - 1) / 2 for x in range(size)], dtype=torch.float32)
        x, y = torch.meshgrid(coords, coords)
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / torch.sum(kernel)
        return kernel



class udkws_atcat_phonecross(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecross, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder_pad(d_model = 100)


        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''
        unsample_list = []
        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)
        if comphn is not None:
            com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)

        # out = l_all
        anco_text_p = self.text_adp2(anco_text_p) 
        com_speech_p = self.audio_adp2(com_speech_p.float())
        cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, com_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(com_speech_p.shape[0]):
            len_anco = mask_textan[i].sum()
            len_com = com_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(40, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4



        # speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        # mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        # speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(pad_anchor_speech) 
        # pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        # out_phone = self.dense_phone(phone_brach) 

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        out = self.gru(pattern_p)
        out = self.dense(out)

        return out, claploss



class udkws_atcat_phonecross2(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecross2, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder_pad(d_model = 100)


        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)
        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]
        '''
        unsample_list = []
        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)

        com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)


        ################   save MA  ##########################
        # fea_ap = exatphone_poolonlya(com_speech_p, anco_mask, an_path)
        # caudio_phnfea, cphn_text = fea_ap
        # caudio_phnfea = self.audio_adp2(caudio_phnfea)
        # for k in range(len(cphn_text)):
        #     new_phone_tmp = caudio_phnfea[k].unsqueeze(0)
        #     old_phone = my_MA[self.idx2p[int(cphn_text[k])]]
        #     thta = 0.8
        #     new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
        #     my_MA[self.idx2p[int(cphn_text[k])]] = new_phone
        ################   save MA  ##########################


        
        # out = l_all
        anco_text_p = self.text_adp2(anco_text_p) 
        com_speech_p = self.audio_adp2(com_speech_p.float())
        cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, com_mask)

        ##################  MA   ####################
        MA_phone_list = []
        for k in range(len(ang2p)):
            phonetmp = []
            for phnname in ang2p[k]:
                phonetmp.append(my_MA[self.idx2p[int(phnname)]])
            batch_phone_MA = pad_sequence(phonetmp, batch_first=True, padding_value=0.0).squeeze(1) #10080,4
            # expanded_tensor = F.pad(batch_phone_MA, (0, 0, 0, 40-len(ang2p[k])))
            MA_phone_list.append(batch_phone_MA)
        phone_MA_embd = pad_sequence(MA_phone_list, batch_first=True, padding_value=0.0).squeeze(1)#10080,4
        MA_cosine_sim_matrix = calculate_cross(phone_MA_embd, com_speech_p, mask_textan, com_mask)

        cosine_sim_matrix = cosine_sim_matrix  + MA_cosine_sim_matrix
        ##################  MA   ####################


        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(com_speech_p.shape[0]):
            len_anco = mask_textan[i].sum()
            len_com = com_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(40, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4



        # speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        # mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        # speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(pad_anchor_speech) 
        # pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        # out_phone = self.dense_phone(phone_brach) 

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        out = self.gru(pattern_p)
        out = self.dense(out)

        return out, my_MA, claploss


class udkws_atcat_phonecrossaug(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecrossaug, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder_pad(d_model = 100)


        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)
        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]
        '''
        unsample_list = []
        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)

        com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)


        


        ################   save MA  ##########################
        # fea_ap = exatphone_poolonlya(com_speech_p, anco_mask, an_path)
        # caudio_phnfea, cphn_text = fea_ap
        # caudio_phnfea = self.audio_adp2(caudio_phnfea)
        # for k in range(len(cphn_text)):
        #     new_phone_tmp = caudio_phnfea[k].unsqueeze(0)
        #     old_phone = my_MA[self.idx2p[int(cphn_text[k])]]
        #     thta = 0.8
        #     new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
        #     my_MA[self.idx2p[int(cphn_text[k])]] = new_phone
        ################   save MA  ##########################

        if self.training:
            com_speech_p = self.audio_adp2(com_speech_p.float())
            anco_text_p = self.text_adp2(anco_text_p) 
            com_speech_p, com_mask, anco_text_p, mask_textan, label = augphone_ma(com_speech_p, com_mask, com_path, anco_text_p, mask_textan, label, my_MA) 
        else:
            com_speech_p = self.audio_adp2(com_speech_p.float())
            anco_text_p = self.text_adp2(anco_text_p) 


        
        # out = l_all
        # anco_text_p = self.text_adp2(anco_text_p) 
        # com_speech_p = self.audio_adp2(com_speech_p.float())
        cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, com_mask)

        ##################  MA   ####################
        # MA_phone_list = []
        # for k in range(len(ang2p)):
        #     phonetmp = []
        #     for phnname in ang2p[k]:
        #         phonetmp.append(my_MA[self.idx2p[int(phnname)]])
        #     batch_phone_MA = pad_sequence(phonetmp, batch_first=True, padding_value=0.0).squeeze(1) #10080,4
        #     # expanded_tensor = F.pad(batch_phone_MA, (0, 0, 0, 40-len(ang2p[k])))
        #     MA_phone_list.append(batch_phone_MA)
        # phone_MA_embd = pad_sequence(MA_phone_list, batch_first=True, padding_value=0.0).squeeze(1)#10080,4
        # MA_cosine_sim_matrix = calculate_cross(phone_MA_embd, com_speech_p, mask_textan, com_mask)

        # cosine_sim_matrix = cosine_sim_matrix  + MA_cosine_sim_matrix
        ##################  MA   ####################


        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(com_speech_p.shape[0]):
            len_anco = mask_textan[i].sum()
            len_com = com_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(40, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4



        # speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        # mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        # speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(pad_anchor_speech) 
        # pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        # out_phone = self.dense_phone(phone_brach) 

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        out = self.gru(pattern_p)
        out = self.dense(out)

        return out, my_MA, claploss, label


class udkws_atcat_phonecrossaug_google(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecrossaug_google, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder_pad(d_model = 100)

        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)
        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, com_path, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]
        '''
        unsample_list = []
        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        # anco_speech_p = self.audio_adp(anco_speech)
        if self.training:
            com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            # fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            # if fea != 1:
            #     audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
            #     l_all = l1
            #     audio_phnfea = audio_phnfea1
            #     phn_text_embd2 = phn_text_embd21
            #     phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            # if fea2 !=1 and fea != 1:
            #     audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
            #     phn_text = torch.cat((phn_text1, phn_text2), dim=0)
            #     phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
            #     l_all = l1 + l2

            if fea2 == 1:
                return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)


        


        ################   save MA  ##########################
        # fea_ap = exatphone_poolonlya(com_speech_p, anco_mask, an_path)
        # caudio_phnfea, cphn_text = fea_ap
        # caudio_phnfea = self.audio_adp2(caudio_phnfea)
        # for k in range(len(cphn_text)):
        #     new_phone_tmp = caudio_phnfea[k].unsqueeze(0)
        #     old_phone = my_MA[self.idx2p[int(cphn_text[k])]]
        #     thta = 0.8
        #     new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
        #     my_MA[self.idx2p[int(cphn_text[k])]] = new_phone
        ################   save MA  ##########################

        if self.training:
            com_speech_p = self.audio_adp2(com_speech_p.float())
            anco_text_p = self.text_adp2(anco_text_p) 

            # com_speech_p, com_mask, anco_text_p, mask_textan, label = augphone_ma(com_speech_p, com_mask, com_path, anco_text_p, mask_textan, label, my_MA) 
        else:
            com_speech_p = self.audio_adp2(com_speech_p.float())
            anco_text_p = self.text_adp2(anco_text_p) 

        
        # out = l_all
        # anco_text_p = self.text_adp2(anco_text_p) 
        # com_speech_p = self.audio_adp2(com_speech_p.float())
        cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, com_mask)

        ##################  MA   ####################
        # MA_phone_list = []
        # for k in range(len(ang2p)):
        #     phonetmp = []
        #     for phnname in ang2p[k]:
        #         phonetmp.append(my_MA[self.idx2p[int(phnname)]])
        #     batch_phone_MA = pad_sequence(phonetmp, batch_first=True, padding_value=0.0).squeeze(1) #10080,4
        #     # expanded_tensor = F.pad(batch_phone_MA, (0, 0, 0, 40-len(ang2p[k])))
        #     MA_phone_list.append(batch_phone_MA)
        # phone_MA_embd = pad_sequence(MA_phone_list, batch_first=True, padding_value=0.0).squeeze(1)#10080,4
        # MA_cosine_sim_matrix = calculate_cross(phone_MA_embd, com_speech_p, mask_textan, com_mask)

        # cosine_sim_matrix = cosine_sim_matrix  + MA_cosine_sim_matrix
        ##################  MA   ####################


        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(com_speech_p.shape[0]):
            len_anco = mask_textan[i].sum()
            len_com = com_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(40, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4



        # speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        # mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        # speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(pad_anchor_speech) 
        # pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        # out_phone = self.dense_phone(phone_brach) 

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        out = self.gru(pattern_p)
        out = self.dense(out)

        return out, my_MA, claploss, label




class udkws_atcat_phonecrossaug_decoder(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecrossaug_decoder, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder_pad(d_model = 100)


        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)
        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path, my_MA, my_MAtxt):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]
        '''
        B, _, _ = anco_speech.shape
        unsample_list = []
        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)

        com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)


        


        ################   save MA  ##########################
        # fea_ap = exatphone_poolonlya(com_speech_p, anco_mask, an_path)
        # caudio_phnfea, cphn_text = fea_ap
        # caudio_phnfea = self.audio_adp2(caudio_phnfea)
        # for k in range(len(cphn_text)):
        #     new_phone_tmp = caudio_phnfea[k].unsqueeze(0)
        #     old_phone = my_MA[self.idx2p[int(cphn_text[k])]]
        #     thta = 0.8
        #     new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
        #     my_MA[self.idx2p[int(cphn_text[k])]] = new_phone
        ################   save MA  ##########################

        if self.training:
            com_speech_p = self.audio_adp2(com_speech_p.float())
            anco_text_p = self.text_adp2(anco_text_p) 
            com_speech_p, com_mask, anco_text_p, mask_textan, label = augphone_ma(com_speech_p, com_mask, com_path, anco_text_p, mask_textan, label, my_MA) 
        else:
            com_speech_p = self.audio_adp2(com_speech_p.float())
            anco_text_p = self.text_adp2(anco_text_p) 


        
        # out = l_all
        # anco_text_p = self.text_adp2(anco_text_p) 
        # com_speech_p = self.audio_adp2(com_speech_p.float())
        cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, com_mask)
        cosine_sim_matrix_tmp = cosine_sim_matrix.clone()

        ##################  MA   ####################
        # UW UH0 OY2 
        MA_phone_list = [] 
        oov_list = []
        for k in range(len(ang2p)):
            if 59 in ang2p[k] or 62 in ang2p[k] or 52 in ang2p[k]:
                oov_list.append(k)
                batch_phone_MA = torch.zeros(40,128).cuda()
                MA_phone_list.append(batch_phone_MA)
                continue
            phonetmp = []
            for phnname in ang2p[k]:
                phonetmp.append(my_MAtxt[self.idx2p[int(phnname)]])
            batch_phone_MA = pad_sequence(phonetmp, batch_first=True, padding_value=0.0).squeeze(1) #10080,4
            # expanded_tensor = F.pad(batch_phone_MA, (0, 0, 0, 40-len(ang2p[k])))
            MA_phone_list.append(batch_phone_MA)
        phone_MA_embd = pad_sequence(MA_phone_list, batch_first=True, padding_value=0.0).squeeze(1)#10080,4
        # MA_cosine_sim_matrix = calculate_cross(phone_MA_embd, com_speech_p, mask_textan, com_mask)
        MA_cosine_sim_matrix = calculate_cross(phone_MA_embd, com_speech_p[:B], mask_textan[:B], com_mask[:B])
        
        cosine_sim_matrix[:B] = (cosine_sim_matrix[:B]  + MA_cosine_sim_matrix) / 2

        if len(oov_list) != 0:
            for idx in oov_list:
                cosine_sim_matrix[int(idx)] =cosine_sim_matrix_tmp[int(idx)]
        ##################  MA   ####################


        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(com_speech_p.shape[0]):
            len_anco = mask_textan[i].sum()
            len_com = com_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(40, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4



        # speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        # mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        # speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(pad_anchor_speech) 
        # pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        # out_phone = self.dense_phone(phone_brach) 

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        out = self.gru(pattern_p)
        out = self.dense(out)

        return out, my_MA, claploss, label






class udkws_atcat_phonecrossaugclass3(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecrossaugclass3, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder_pad(d_model = 100)


        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)

        self.transformer_en3 = Transformer_encoder_pad(d_model = 100)
        self.gru3 = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        self.dense3 = nn.Linear(100, 3)


        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, label_3, an_path, com_path, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]
        '''
        unsample_list = []
        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)

        com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        out3 = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2

            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)


        


        ################   save MA  ##########################
        # fea_ap = exatphone_poolonlya(com_speech_p, anco_mask, an_path)
        # caudio_phnfea, cphn_text = fea_ap
        # caudio_phnfea = self.audio_adp2(caudio_phnfea)
        # for k in range(len(cphn_text)):
        #     new_phone_tmp = caudio_phnfea[k].unsqueeze(0)
        #     old_phone = my_MA[self.idx2p[int(cphn_text[k])]]
        #     thta = 0.8
        #     new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
        #     my_MA[self.idx2p[int(cphn_text[k])]] = new_phone
        ################   save MA  ##########################

        if self.training:
            com_speech_p = self.audio_adp2(com_speech_p.float())
            anco_text_p = self.text_adp2(anco_text_p) 
            # com_speech_p, com_mask, anco_text_p, mask_textan, label, label_3 = augphone_ma_3class(com_speech_p, com_mask, com_path, anco_text_p, mask_textan, label, label_3, my_MA) 

            com_speech_p, com_mask, anco_text_p, mask_textan, label, label_3 = augphone_ma_3class_3hard(com_speech_p, com_mask, com_path, anco_text_p, mask_textan, label, label_3, my_MA) 


            
        else:
            com_speech_p = self.audio_adp2(com_speech_p.float())
            anco_text_p = self.text_adp2(anco_text_p) 


        
        # out = l_all
        # anco_text_p = self.text_adp2(anco_text_p) 
        # com_speech_p = self.audio_adp2(com_speech_p.float())
        cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, com_mask)

        ##################  MA   ####################
        # MA_phone_list = []
        # for k in range(len(ang2p)):
        #     phonetmp = []
        #     for phnname in ang2p[k]:
        #         phonetmp.append(my_MA[self.idx2p[int(phnname)]])
        #     batch_phone_MA = pad_sequence(phonetmp, batch_first=True, padding_value=0.0).squeeze(1) #10080,4
        #     # expanded_tensor = F.pad(batch_phone_MA, (0, 0, 0, 40-len(ang2p[k])))
        #     MA_phone_list.append(batch_phone_MA)
        # phone_MA_embd = pad_sequence(MA_phone_list, batch_first=True, padding_value=0.0).squeeze(1)#10080,4
        # MA_cosine_sim_matrix = calculate_cross(phone_MA_embd, com_speech_p, mask_textan, com_mask)

        # cosine_sim_matrix = cosine_sim_matrix  + MA_cosine_sim_matrix
        ##################  MA   ####################


        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(com_speech_p.shape[0]):
            len_anco = mask_textan[i].sum()
            len_com = com_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(40, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4


        if self.training:
            pattern_p3 = self.transformer_en3(pad_anchor_speech) 
            out3 = self.gru3(pattern_p3)
            out3 = self.dense3(out3)

        pattern_p = self.transformer_en(pad_anchor_speech) 

        out = self.gru(pattern_p)
        out = self.dense(out)

        return out, out3, my_MA, claploss, label, label_3



class udkws_atcat_phonecrossaug_tMA(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecrossaug_tMA, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder_pad(d_model = 100)


        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(100, 1)
        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]
        '''
        unsample_list = []
        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)

        com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)


        


            ################   save MA  ##########################
            fea_ap = exatphone_poolonlya(com_speech_p, anco_mask, an_path)
            caudio_phnfea, cphn_text = fea_ap
            caudio_phnfea = self.audio_adp2(caudio_phnfea)
            for k in range(len(cphn_text)):
                new_phone_tmp = caudio_phnfea[k].unsqueeze(0)
                old_phone = my_MA[self.idx2p[int(cphn_text[k])]]
                thta = 0.8
                new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
                my_MA[self.idx2p[int(cphn_text[k])]] = new_phone
            ################   save MA  ##########################

        if self.training:
            com_speech_p = self.audio_adp2(com_speech_p.float())
            anco_text_p = self.text_adp2(anco_text_p) 
            # com_speech_p, com_mask, anco_text_p, mask_textan, label = augphone_ma(com_speech_p, com_mask, com_path, anco_text_p, mask_textan, label, my_MA) 
            # com_speech_p, com_mask, anco_text_p, mask_textan, label = augphone_ma(com_speech_p, com_mask, com_path, anco_text_p, mask_textan, label, my_MA) 

        else:
            com_speech_p = self.audio_adp2(com_speech_p.float())
            anco_text_p = self.text_adp2(anco_text_p) 


        
        # out = l_all
        # anco_text_p = self.text_adp2(anco_text_p) 
        # com_speech_p = self.audio_adp2(com_speech_p.float())
        cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, com_mask)
        if self.training == False:
            ##################  MA   ####################
            MA_phone_list = []
            for k in range(len(ang2p)):
                phonetmp = []
                for phnname in ang2p[k]:
                    phonetmp.append(my_MA[self.idx2p[int(phnname)]])
                batch_phone_MA = pad_sequence(phonetmp, batch_first=True, padding_value=0.0).squeeze(1) #10080,4
                # expanded_tensor = F.pad(batch_phone_MA, (0, 0, 0, 40-len(ang2p[k])))
                MA_phone_list.append(batch_phone_MA)
            phone_MA_embd = pad_sequence(MA_phone_list, batch_first=True, padding_value=0.0).squeeze(1)#10080,4
            # MA_cosine_sim_matrix = calculate_cross(phone_MA_embd, com_speech_p[:16], mask_textan[:16], com_mask[:16])
            MA_cosine_sim_matrix = calculate_cross(phone_MA_embd, com_speech_p, mask_textan, com_mask)


            cosine_sim_matrix = cosine_sim_matrix  + MA_cosine_sim_matrix * 0.5
            ##################  MA   ####################


        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(com_speech_p.shape[0]):
            len_anco = mask_textan[i].sum()
            len_com = com_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(40, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4



        # speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        # mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        # speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(pad_anchor_speech) 
        # pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        # out_phone = self.dense_phone(phone_brach) 

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        out = self.gru(pattern_p)
        out = self.dense(out)

        return out, my_MA, claploss, label



class udkws_atcat_phonecrossatt(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phonecrossatt, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.criterion = ContrastiveLoss_mask()
        self.transformer = Transformer_cross(embed_size = 40, num_layers = 1, num_heads = 1, ff_hidden_size = 40, dropout = 0.1)
        # self.transformer_en = Transformer_encoder_pad(d_model = 100)


        # self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(40, 1)
        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.phonemes = ["<pad>", ] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', ' '] 
        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''
        unsample_list = []
        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)

        com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            for k in range(len(phn_text)):
                new_phone_tmp = anco_speech_p2[k].unsqueeze(0)
                old_phone = my_MA[self.idx2p[int(phn_text[k])]]
                thta = 0.8
                new_phone = thta * new_phone_tmp + (1 - thta) * old_phone
                my_MA[self.idx2p[int(phn_text[k])]] = new_phone

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)

        # out = l_all
        anco_text_p = self.text_adp2(anco_text_p) 
        com_speech_p = self.audio_adp2(com_speech_p.float())


        
        cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, com_mask)

        #########################    MA    #################
        # if self.training == False:
        #     MA_phone_list = []
        #     for k in range(len(ang2p)):
        #         phonetmp = []
        #         for phnname in ang2p[k]:
        #             phonetmp.append(my_MA[self.idx2p[int(phnname)]])
        #         batch_phone_MA = pad_sequence(phonetmp, batch_first=True, padding_value=0.0).squeeze(1) #10080,4
        #         # expanded_tensor = F.pad(batch_phone_MA, (0, 0, 0, 40-len(ang2p[k])))
        #         MA_phone_list.append(batch_phone_MA)
        #     phone_MA_embd = pad_sequence(MA_phone_list, batch_first=True, padding_value=0.0).squeeze(1)#10080,4
        #     MA_cosine_sim_matrix = calculate_cross(phone_MA_embd, com_speech_p, mask_textan, com_mask)

        #     cosine_sim_matrix2 = cosine_sim_matrix  + MA_cosine_sim_matrix
        #########################    MA    #################

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(com_speech_p.shape[0]):
            len_anco = mask_textan[i].sum()
            len_com = com_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(40, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4



        # speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        # mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        # speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)
        # max_corss_speech = pad_anchor_speech.mean(dim = 1).unsqueeze(1)
        # max_corss_text = pad_anchor_speech.max(dim = 1).values.unsqueeze(1)
        # pattern_pt = self.transformer(pad_anchor_speech, pad_anchor_speech, max_corss_text)

        max_corss_speech = pad_anchor_speech.max(dim = 2).values.unsqueeze(1)
        pattern_p = self.transformer(pad_anchor_speech.transpose(1,2), pad_anchor_speech.transpose(1,2), max_corss_speech)

        # pattern_p = self.transformer_en(pad_anchor_speech) 
        # pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        # phone_brach = pattern_p[:,100:,:] #b, 8,128 
        
        # out_phone = self.dense_phone(phone_brach) 

        # paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        # logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        # loss = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        out = pattern_p.squeeze(1)
        # out = self.gru(pattern_p)
        out = self.dense(out)

        return out, claploss, my_MA



class udkws_atcat_phone_catcs(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phone_catcs, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder_pad(d_model = 100)


        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        # self.dense = nn.Linear(100, 1)
        self.dense = nn.Linear(228, 1)

        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()

        self.transformer_cat = Transformer_encoder()

        self.grucat = GRUFCModel_mask(input_dim=128, hidden_dim=128, output_dim=128)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        # self.densecat = nn.Linear(128, 1)
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''
        unsample_list = []
        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)
        if comphn is not None:
            com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)
        ################################   aug   ##################
        # if self.training:
        #     com_speech_p = self.audio_adp2(com_speech_p.float())
        #     anco_text_p = self.text_adp2(anco_text_p) 
        #     com_speech_p, com_mask, anco_text_p, mask_textan, label = augphone_ma(com_speech_p, com_mask, com_path, anco_text_p, mask_textan, label, my_MA) 
        # else:
        #     com_speech_p = self.audio_adp2(com_speech_p.float())
        #     anco_text_p = self.text_adp2(anco_text_p) 
        ################################   aug   ##################
        

        # out = l_all
        # anco_text_p = self.text_adp2(anco_text_p) 
        # com_speech_p = self.audio_adp2(com_speech_p.float())

        speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))
        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)
        pattern_pcat = self.transformer_cat(speech_text, mask_cross) 
        pattern_pcat = pattern_pcat * speech_text_mask
        outcat = self.grucat(pattern_pcat, speech_text_mask)



        cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, com_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(com_speech_p.shape[0]):
            len_anco = mask_textan[i].sum()
            len_com = com_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(40, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4


        pattern_p = self.transformer_en(pad_anchor_speech) 
        # pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        outcs = self.gru(pattern_p)
        out = torch.cat((outcs, outcat), dim = 1)
        out = self.dense(out)

        return out, claploss



class udkws_atcat_phone_catcsaug(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_phone_catcsaug, self).__init__()
        self.text_adp = Projection(input_dim=256, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.text_adp2 = Projection(input_dim=128, output_dim=128)
        self.audio_adp2 = Projection(input_dim=128, output_dim=128)

        self.kernel_size = 5
        self.sigma = 1.0
        self.gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma).view(1, 1, self.kernel_size, self.kernel_size).cuda()

        self.criterion = ContrastiveLoss_mask()

        self.transformer_en = Transformer_encoder_pad(d_model = 100)


        self.gru = GRUFCModel(input_dim=100, hidden_dim=100, output_dim=100)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        # self.dense = nn.Linear(100, 1)
        self.dense = nn.Linear(228, 1)

        # self.dense_phone = nn.Linear(128, 1)
        # self.bce_loss = nn.BCEWithLogitsLoss()

        self.transformer_cat = Transformer_encoder()

        self.grucat = GRUFCModel_mask(input_dim=128, hidden_dim=128, output_dim=128)
        # # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        # self.densecat = nn.Linear(128, 1)
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, ang2p, anphn, comg2p, comphn, mask_textan, mask_textcom, label, an_path, com_path, my_MA):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''
        unsample_list = []
        mask_text = mask_textan

        anco_text_p = self.text_adp(anphn.float()) 
        anco_speech_p = self.audio_adp(anco_speech)
        if comphn is not None:
            com_text_p = self.text_adp(comphn.float()) 
        com_speech_p = self.audio_adp(com_speech)
        claploss = 0
        # cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, comparison_mask)
        # com_text_p = self.text_adp(comphn.float())
        # com_speech_p = self.audio_adp(com_speech)
        if self.training:
            fea = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            fea2 = exatphone_pool(com_speech_p, com_mask, com_path, comg2p, com_text_p, mask_textcom) 
            
            if fea != 1:
                audio_phnfea1, phn_text1, phn_text_embd21, l1 = fea
                l_all = l1
                audio_phnfea = audio_phnfea1
                phn_text_embd2 = phn_text_embd21
                phn_text = phn_text1
            if fea2 !=1:
                audio_phnfea2, phn_text2, phn_text_embd22, l2 = fea2
                l_all = l2
                audio_phnfea = audio_phnfea2
                phn_text_embd2 = phn_text_embd22
                phn_text = phn_text2


            if fea2 !=1 and fea != 1:
                audio_phnfea = torch.cat((audio_phnfea1, audio_phnfea2), dim=0)
                phn_text = torch.cat((phn_text1, phn_text2), dim=0)
                phn_text_embd2 = torch.cat((phn_text_embd21, phn_text_embd22), dim=0)
                l_all = l1 + l2

            if fea2 == 1 and fea == 1:
                return 0
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 
            # audio_phnfea, phn_text, phn_text_embd2 = exatphone_pool(anco_speech_p, anco_mask, an_path, ang2p, anco_text_p, mask_textan) 

            anco_text_p2 = self.text_adp2(phn_text_embd2) 
            anco_speech_p2 = self.audio_adp2(audio_phnfea.float())

            claploss = self.criterion(anco_speech_p2, anco_text_p2, phn_text)
        ################################   aug   ##################
        if self.training:
            com_speech_p = self.audio_adp2(com_speech_p.float())
            anco_text_p = self.text_adp2(anco_text_p) 
            com_speech_p, com_mask, anco_text_p, mask_textan, label = augphone_ma(com_speech_p, com_mask, com_path, anco_text_p, mask_textan, label, my_MA) 
        else:
            com_speech_p = self.audio_adp2(com_speech_p.float())
            anco_text_p = self.text_adp2(anco_text_p) 
        ################################   aug   ##################
        

        # out = l_all
        # anco_text_p = self.text_adp2(anco_text_p) 
        # com_speech_p = self.audio_adp2(com_speech_p.float())

        speech_text_mask = torch.cat((com_mask, mask_textan), dim=1)
        mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))
        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)
        pattern_pcat = self.transformer_cat(speech_text, mask_cross) 
        pattern_pcat = pattern_pcat * speech_text_mask
        outcat = self.grucat(pattern_pcat, speech_text_mask)



        cosine_sim_matrix = calculate_cross(anco_text_p, com_speech_p, mask_textan, com_mask)

        # cosine_sim_matrix = calculate_cross(anco_speech_p, com_speech_p)
        cosine_sim_matrix = cosine_sim_matrix.unsqueeze(1)

        for i in range(com_speech_p.shape[0]):
            len_anco = mask_textan[i].sum()
            len_com = com_mask[i].sum()
            cosine_sim_matrix_tmp = cosine_sim_matrix[i, 0, :int(len_anco), :int(len_com)].unsqueeze(0).unsqueeze(1)

            # up = self.upsample(cosine_sim_matrix)
            # dec = self.deconv(cosine_sim_matrix)

            upsampled_tensor = F.interpolate(cosine_sim_matrix_tmp, size=(40, 100), mode='bilinear', align_corners=True)
            smoothed_tensor = F.conv2d(upsampled_tensor, self.gaussian_kernel, padding=self.kernel_size // 2).squeeze(0)
            unsample_list.append(smoothed_tensor)
        pad_anchor_speech = pad_sequence(unsample_list, batch_first=True, padding_value=0.0).squeeze(1)  #10080,4


        pattern_p = self.transformer_en(pad_anchor_speech) 
        # pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        outcs = self.gru(pattern_p)
        out = torch.cat((outcs, outcat), dim = 1)
        out = self.dense(out)

        return  out, my_MA, claploss, label



class udkws_atcat_word2(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_word2, self).__init__()
        self.bert_encoder = BERT_encoder() 
        self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel(input_dim=128, hidden_dim=128, output_dim=128)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, label, path):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''
        # print("com_text", com_text)
        anco_text_e, mask_text = self.bert_encoder(anco_text, com_text, path) 
        anco_text_e = anco_text_e.cuda()
        mask_text = mask_text.unsqueeze(-1).float()

        anco_text_p = self.text_adp(anco_text_e) 

        anco_speech_p = self.audio_adp(anco_speech)

        com_speech_p = self.audio_adp(com_speech)

        speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask

        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        out = self.gru(pattern_p)
        out = self.dense(out)


        return out



class udkws_atcatdata(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcatdata, self).__init__()
        self.bert_encoder = BERT_encoder() 
        self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel(input_dim=128, hidden_dim=128, output_dim=128)
        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, comparison_text, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''
        # anco_text.extend(comparison_text)
        anco_text_t, anmask_text = self.bert_encoder(anco_text)
        com_text_t, comask_text = self.bert_encoder(comparison_text)
        anco_text_e = torch.cat((anco_text_t, com_text_t), dim=0)
        mask_text = torch.cat((anmask_text, comask_text), dim=0)


        anco_text_e = anco_text_e.cuda()
        mask_text = mask_text.unsqueeze(-1).float()

        anco_text_p = self.text_adp(anco_text_e) 

        anco_speech_p = self.audio_adp(anco_speech)

        com_speech_p = self.audio_adp(com_speech)

        speechan = torch.cat((anco_speech_p, anco_speech_p, com_speech_p, com_speech_p), dim=0)
        maskan = torch.cat((anco_mask, anco_mask, com_mask, com_mask), dim=0)

        textan = torch.cat((anco_text_p, anco_text_p), dim=0)
        mask_text = torch.cat((mask_text, mask_text), dim=0)

        speech_text_mask = torch.cat((maskan, mask_text), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask, speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((speechan, textan), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask

        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 

        out = self.gru(pattern_p)
        out = self.dense(out)
        label_ture = torch.ones(16).cuda()
        label = torch.cat((label_ture, label, label, label_ture), dim=0)

        return out, label




class udkws_atcat_aword(nn.Module):
    def __init__(self, embedding = 128):
        super(udkws_atcat_aword, self).__init__()
        self.bert_encoder = BERT_encoder_word() 
        self.text_adp = Projection(input_dim=768, output_dim=128)
        self.audio_adp = Projection(input_dim=384, output_dim=128)

        self.transformer_en = Transformer_encoder()

        self.gru = GRUFCModel(input_dim=128, hidden_dim=128, output_dim=128)
        self.gru_a = GRUFCModel(input_dim=128, hidden_dim=128, output_dim=128)

        # self.dis = discriminator_c(input_size = 128, num_channels = 128)
        self.dense = nn.Linear(128, 1)
        self.dense_phone = nn.Linear(128, 1)
        self.dense_audio = nn.Linear(128, 1)

        self.bce_loss = nn.BCEWithLogitsLoss()

    
    def forward(self, anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, label):
        '''
        anco_speech [16, 100, 384]
        anco_text_e [16, 10, 768]
        adp
        anco_speech [16, 100, 128]
        anco_text_e [16, 10, 128]

        '''
        anco_text_e, mask_text, phone_label = self.bert_encoder(anco_text, com_text) 
        anco_text_e = anco_text_e.cuda()
        mask_text = mask_text.unsqueeze(-1).float()

        anco_text_p = self.text_adp(anco_text_e) 
        anco_speech_p = self.audio_adp(anco_speech)
        com_speech_p = self.audio_adp(com_speech)

        speech_text_mask = torch.cat((com_mask, mask_text), dim=1)
        # mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2)).bool()
        mask_cross = torch.bmm(speech_text_mask,speech_text_mask.transpose(1, 2))

        speech_text = torch.cat((com_speech_p, anco_text_p), dim=1)

        pattern_p = self.transformer_en(speech_text, mask_cross) 
        pattern_p = pattern_p * speech_text_mask  # b, 108, 128

        phone_brach = pattern_p[:,100:,:] #b, 8,128
        
        out_phone = self.dense_phone(phone_brach)

        paired_label = torch.masked_select(phone_label, mask_text.bool().squeeze(-1)).unsqueeze(1)
        logits = torch.masked_select(out_phone.squeeze(-1), mask_text.bool().squeeze(-1)).unsqueeze(1)
        loss_t = self.bce_loss(logits.float(), paired_label.float())#8384,1920
        
        
        # output, hidden = self.gru(pattern_p)
        # out = hidden[-1]
        # gru_out, _ = self.gru(pattern_p)
        # out = gru_out[:, -1, :] 
        a_out = pattern_p[:,:100,:]
        a_out = self.gru_a(a_out)
        out_a = self.dense_audio(a_out)
        if len(label.shape) == 1:
            loss_a = self.bce_loss(out_a.float(), label.unsqueeze(-1).float())#8384,1920
        else:
            loss_a = self.bce_loss(out_a.float(), label.float()) 

        loss = loss_a + loss_t


        out = self.gru(pattern_p)
        out = self.dense(out)

        return out, loss





# text = ['is a city nice','length']
# text2 = ['is a nice','length']

# # text = ['is a city']
# anco_speech = torch.randn(2, 100, 384).cuda()
# anco_mask = torch.randn(2, 100, 1).cuda()
# com_speech = torch.randn(2, 100, 384).cuda()
# com_mask = torch.randn(2, 100, 1).cuda()
# anco_text = text = ['is a city nice','length']
# com_text = text2 = ['is a nice','length']
# label = torch.randn(2, 1)



# # bert = BERT_encoder_word() 
# # output_text = bert(text, text2) 
# # print(output_text.shape)

# udmodel = udkws_atcat_word()
# udmodel = udmodel.cuda()
# output_text = udmodel(anco_speech, anco_mask, com_speech, com_mask, anco_text, com_text, label) 

#tensor([-0.2273,  0.0171,  0.0608, -0.0941, -0.1123, -0.1212, -0.1630, -0.1648],
    #    device='cuda:0')
    #  2 tensor([-0.1920, -0.1232, -0.5403, -0.3936,  0.6145, -0.0872, -0.0627, -0.0428,
        # -0.1531, -0.0739], device='cuda:0')
# input_embedding = torch.randn(2, 1500, 384)
# conv_pool_encoder = udkws()

# output_embedding = conv_pool_encoder(input_embedding,text,input_embedding)  # (8, 128, T)

# print(output_embedding.shape)  # 输出: (8, T, 128)

