import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.io as sio
import numpy as np
import sys

class KWSIflytekLoss(nn.Module):
    def __init__(self, unitList_student, unitList_teacher, train_node_None):
        super(KWSIflytekLoss, self).__init__()
        self.celoss           = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.kld_critirion    = nn.KLDivLoss(reduction='none')
        if unitList_student != None and unitList_teacher != None:
            unitList_student  = unitList_student.split(',')
            unitList_student  = Variable( torch.tensor(np.array( [int(x) for x in unitList_student] )).reshape(1, -1) ).detach()
            unitList_teacher  = unitList_teacher.split(',')
            unitList_teacher  = Variable( torch.tensor(np.array( [int(x) for x in unitList_teacher] )).reshape(1, -1) ).detach()
        self.unitList_student = unitList_student
        self.unitList_teacher = unitList_teacher
        self.train_node_None  = train_node_None

    def func_hardSample(self, x, label, sampleScore, focalGama):
        gather = label.clone().long()
        gather[gather<0] = 0

        probs = x.clone()
        probs = F.softmax(probs, dim=1)
        probs = torch.gather(probs, 1, gather.unsqueeze(1)).squeeze(1)

        # hardSample标签
        mask = probs.clone()
        matrix_neg = torch.ones_like(mask) * (-2)
        label_hardSample = torch.where(mask<sampleScore, label, matrix_neg.long()) if sampleScore < 1.0 and sampleScore > 0.0 else label
        # focal系数
        focal_weight = torch.pow((1-probs), focalGama) if focalGama > 1.0 else torch.new_ones(probs)
        return label_hardSample, focal_weight

    def forward(self, logit_sudent, logit_teacher, kld_gamma, frames_label, num_classes, nmod, is_trainNone, is_hardSample=False, sampleScore=1.0, focalGama=1.0):
        '''
            logit_s shape: N, 3003, T/nmod
            logit_t shape: N, 3003, T/nmod
            frames_label shape: N, T
        '''
        out_loss   = torch.tensor(0)
        frames_label = frames_label.reshape(-1, 1)
        logit_sudent = logit_sudent.reshape(-1, logit_sudent.size(-1))
        if logit_teacher != None:
            logit_teacher = logit_teacher.reshape(-1, logit_teacher.size(-1))
        
        # 有监督正例loss
        target     = frames_label.clone().flatten()
        target[ target < 0 ] = -1
        target[ target == self.train_node_None ] = -1
        target     = target.long().reshape(-1, nmod)[:,0]
        mask_ones  = torch.ones_like(target)
        mask_zeros = torch.zeros_like(target)

        # 有监督数据难例采样
        mask_weight = mask_ones.clone()
        if is_hardSample:
            target, mask_weight = self.func_hardSample(logit_sudent, target, sampleScore, focalGama)
        
        kld_mask   = torch.where(target>=0, mask_ones, mask_zeros)
        kld_weight = torch.where(target>=0, mask_weight.float(), mask_zeros.float())
        if torch.sum(kld_mask) > 0:
            if kld_gamma<1.0 and kld_gamma>0.0:
                logit_teacher_super = F.softmax(logit_teacher, 1).detach()
                logit_student_super = F.log_softmax(logit_sudent, 1)
                kld_tar             = (1-kld_gamma)*F.one_hot((target*kld_mask).long(), num_classes=num_classes) + kld_gamma*logit_teacher_super
                out_loss            = self.kld_critirion(logit_student_super, kld_tar)
                out_loss            = torch.sum((torch.sum(out_loss, 1)*kld_weight))/torch.sum(kld_mask)
            else:
                target              = target.unsqueeze(-1)
                target              = target.flatten().long()
                logit_super         = logit_sudent.reshape((-1, num_classes))
                out_loss            = self.celoss(logit_super, target)

        # 无监督反例loss
        target_none = frames_label.clone().flatten()
        target_none[ target_none != self.train_node_None ] = -1
        target_none = target_none.long().reshape(-1, nmod)[:,0]
        kld_none_mask = torch.where(target_none>=0, mask_ones, mask_zeros)

        if is_trainNone and torch.sum(kld_none_mask) > 0 and logit_teacher != None and self.unitList_teacher != None and self.unitList_student != None:
            
            if logit_sudent.is_cuda and not self.unitList_student.is_cuda and not self.unitList_teacher.is_cuda:
                self.unitList_student = self.unitList_student.cuda()
                self.unitList_teacher = self.unitList_teacher.cuda()

            N_s, C_s = logit_sudent.size()
            N_t, C_t = logit_teacher.size()

            states_mask_student  = logit_sudent.data.new(N_s, C_s).fill_(1)
            states_mask_student  = Variable(states_mask_student)
            _, ids_sel_student   = torch.broadcast_tensors(logit_sudent[:, :self.unitList_student.size(1)], self.unitList_student)
            states_mask_student.scatter_(1, ids_sel_student.data, 0)

            states_mask_teacher  = logit_teacher.data.new(N_t, C_t).fill_(1)
            states_mask_teacher  = Variable(states_mask_teacher)
            _, ids_sel_teacher   = torch.broadcast_tensors(logit_teacher[:, :self.unitList_teacher.size(1)], self.unitList_teacher)
            states_mask_teacher.scatter_(1, ids_sel_teacher.data, 0)

            logit_sudent_unsuper = F.softmax(logit_sudent, dim=1)
            logit_student_mask   = logit_sudent_unsuper * states_mask_student
            logit_student_mask   = torch.sum(logit_student_mask, dim=1)

            logit_teacher_unsuper = F.softmax(logit_teacher, 1).detach()
            logit_teacher_mask    = logit_teacher_unsuper.clone() * states_mask_teacher
            logit_teacher_mask    = torch.sum(logit_teacher_mask, dim=1)

            kld_none_loss = -(logit_teacher_mask*0.5 + 0.5) * (logit_student_mask.log())
            kld_none_loss = torch.sum(kld_none_loss*kld_none_mask)/torch.sum(kld_none_mask)
            out_loss += kld_none_loss
        return out_loss

class KWSIflytekLoss2(nn.Module):
    def __init__(self, kws_statesList, train_node_None):
        super(KWSIflytekLoss2, self).__init__()
        if kws_statesList != None:
            kws_statesList   = kws_statesList.split(',')
            kws_statesList   = Variable( torch.tensor(np.array( [int(x) for x in kws_statesList] )).reshape(-1)).detach()
        self.kws_statesList  = kws_statesList
        self.train_node_None = train_node_None

    def forward(self, x, frames_label, num_classes, nmod, is_trainNone, sampleScore=1.0, focalGama=1.0, noneAlpha=1.0):
        '''
            x shape: N, 3003, T/nmod
            frames_label shape: N, T
        '''
        out_loss = torch.tensor(0).cuda() if x.is_cuda else torch.tensor(0)
        x = x.reshape(-1, x.size(-1))
        frames_label = frames_label.reshape(-1, 1)
        
        # 正例loss
        target_pos = frames_label.clone().flatten()
        target_pos[ target_pos < 0 ] = -1
        if self.train_node_None != None:
            target_pos[ target_pos == self.train_node_None ] = -1
        target_pos = target_pos.long().reshape(-1, nmod)[:, 0]
        mask_ones  = torch.ones_like(target_pos)
        mask_zeros = torch.zeros_like(target_pos)

        gather  = target_pos.clone().view(-1, 1)
        gather[gather<0] = 0
        probs   = F.softmax(x, dim=1)
        value_p = torch.gather(probs, 1, gather).view(-1)
        log_p   = (value_p+1e-10).log()

        if sampleScore<1.0 and sampleScore>0.0:
            mask_sample = torch.ones_like(target)*(-2)
            target_pos  = torch.where(value_p<sampleScore, target_pos, mask_sample)
        pos_mask   = torch.where(target_pos>=0, mask_ones, mask_zeros)
        pos_weight = torch.pow((1-value_p), focalGama) if focalGama > 1.0 else pos_mask.new_ones(pos_mask.size())
        n_posValid = torch.sum(pos_mask)
        if n_posValid > 0:
            loss_pos = -(pos_mask * pos_weight)*log_p
            out_loss = loss_pos.sum()/n_posValid

        # 反例loss
        loss_neg = torch.tensor(0)
        if is_trainNone:
            max_class  = max(self.train_node_None, num_classes)+1 if self.train_node_None !=None else num_classes+1
            target_tmp = frames_label.clone().long().reshape(-1, nmod)[:, 0]
            target_tmp[ target_tmp<0 ] = max_class
            mask_class = torch.tensor(range(max_class+1))

            if probs.is_cuda:
                self.kws_statesList = self.kws_statesList.cuda()
                mask_class = mask_class.cuda()
            
            mask_class.scatter_(0, self.kws_statesList, -6)
            target_neg = torch.gather(mask_class, 0, target_tmp)
            target_neg[ target_neg==max_class ] = -2
            neg_mask   = torch.where(target_neg>=0, mask_ones, mask_zeros)
            n_negValid = torch.sum(neg_mask)
            if n_negValid > 0 and self.kws_statesList != None:
                n, c = probs.size()
                states_mask = probs.data.new(n, c).fill_(1)
                states_mask = Variable(states_mask)
                _, ids_sel  = torch.broadcast_tensors(probs[:, :self.kws_statesList.size(0)], self.kws_statesList.reshape(1, -1))
                states_mask.scatter_(1, ids_sel.data, 0)
                probs_mask = states_mask*probs
                probs_mask = torch.sum(probs_mask, dim=1)
                log_mask   = (probs_mask+1e-10).log()

                if noneAlpha>0:
                    neg_weight = probs_mask*noneAlpha
                    neg_weight[neg_weight<0.05] = 0.05
                    neg_weight[neg_weight>1.00] = 1.00
                    neg_weight = 1 / (neg_weight+1e-10)
                else:
                    neg_weight = neg_mask.new_ones(neg_mask.size())
                
                loss_neg = -(neg_mask * neg_weight)*log_mask
                out_loss += loss_neg.sum()/n_negValid
        return out_loss

class KWSIflytekLoss3(nn.Module):
    def __init__(self, kws_statesList, node_none, reduction="mean", smooth_alphaPos=0.0, smooth_alphaNeg=0.0, smooth_alphaBce=0.0):
        super(KWSIflytekLoss3, self).__init__()
        self.critirion_cePos = nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction, label_smoothing=smooth_alphaPos)
        self.critirion_ceNeg = nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction, label_smoothing=smooth_alphaNeg)
        self.critirion_ceBce = nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction, label_smoothing=smooth_alphaBce)
        if kws_statesList != None:
            kws_statesList  = kws_statesList.split(',')
            kws_statesList  = Variable( torch.tensor(np.array( [int(x) for x in kws_statesList] )).reshape(-1)).detach()
        self.kws_statesList = kws_statesList
        self.node_none = node_none

    def forward(self, x, frames_label, num_classes, nmod, samplePosScore=1.0, sampleNegScore=1.0, sampleBceScore=1.0):
        '''
            x shape: N, 3003, T/nmod
            frames_label shape: N, T
        '''
  
        x = x.reshape(-1, x.size(-1))
        frames_label = frames_label.reshape(-1, 1)

        target = frames_label.clone().long().flatten()
        target[ target < 0 ] = -1
        if self.node_none != None:
            target[ target == self.node_none ] = -1
        target = target.long().reshape(-1, nmod)[:, 0]

        if self.kws_statesList == None:
            out_loss = self.critirion_cePos(x, target)
        else:
            out_loss = torch.tensor(0.0).cuda() if x.is_cuda else torch.tensor(0.0)

            gather = target.clone().view(-1, 1)
            gather[gather<0] = 0
            probs = F.softmax(x, dim=1)
            class_prob = torch.gather(probs, 1, gather).squeeze(1)

            mask_ones   = torch.ones_like(target)
            mask_zeros  = torch.zeros_like(target)
            mask_ignore = torch.ones_like(target)*(-1)

            max_class  = max(self.node_none, num_classes)+1 if self.node_none !=None else num_classes+1
            target_tmp = frames_label.clone().long().flatten().reshape(-1, nmod)[:, 0]
            target_tmp[ target_tmp<0 ] = max_class
            mask_class = torch.tensor(range(max_class+1))

            if probs.is_cuda:
                self.kws_statesList = self.kws_statesList.cuda()
                mask_class = mask_class.cuda()
            
            mask_class.scatter_(0, self.kws_statesList, -6)
            target_bce = torch.gather(mask_class, 0, target_tmp)
            target_bce[target_bce==max_class] = -1
            target_bce[target_bce>=0] = 1
            target_bce[target_bce==-6] = 0

            # loss pos
            target_pos = torch.where(target_bce==0, target, mask_ignore)
            if samplePosScore<1.0 and samplePosScore>0.0:
                target_pos = torch.where(class_prob<samplePosScore, target_pos, mask_ignore)
            pos_mask   = torch.where(target_pos>=0, mask_ones, mask_zeros)
            n_posValid = torch.sum(pos_mask)
            if n_posValid > 0:
                out_loss += self.critirion_cePos(x, target_pos.detach())

            # loss neg
            target_neg = torch.where(target_bce==1, target, mask_ignore)
            if sampleNegScore<1.0 and sampleNegScore>0.0:
                target_neg = torch.where(class_prob<samplePosScore, target_neg, mask_ignore)
            neg_mask   = torch.where(target_neg>=0, mask_ones, mask_zeros)
            n_negValid = torch.sum(neg_mask)
            if n_negValid > 0:
                out_loss += self.critirion_ceNeg(x, target_neg.detach())

            # loss bce        
            n, c = probs.size()
            states_mask = probs.data.new(n, c).fill_(-1)
            states_mask = Variable(states_mask)
            _, ids_sel  = torch.broadcast_tensors(probs[:, :self.kws_statesList.size(0)], self.kws_statesList.reshape(1, -1))

            states_pos = states_mask.clone().detach()
            states_pos.scatter_(1, ids_sel.data, 1)
            states_pos[states_pos==-1] = 0

            states_neg = states_mask.clone().detach()
            states_neg.scatter_(1, ids_sel.data, 0)
            states_neg[states_neg==-1] = 1

            probs_pos = states_pos*probs
            probs_pos = torch.sum(probs_pos, dim=1)
            probs_neg = states_neg*probs
            probs_neg = torch.sum(probs_neg, dim=1)
            probs_bce = torch.cat( (probs_pos.reshape(-1, 1), probs_neg.reshape(-1, 1)), dim=1)

            if sampleBceScore<1.0 and sampleBceScore>0.0:
                target_bce  = torch.where(probs_neg<sampleNegScore, target_bce, mask_ignore)

            bce_mask   = torch.where(target_bce>=0, mask_ones, mask_zeros)
            n_bceValid = torch.sum(bce_mask)
            if n_bceValid > 0:
                out_loss += self.critirion_ceBce(probs_bce, target_bce.detach())
        return out_loss

class KWSIflytekLoss4(nn.Module):
    def __init__(self, node_none, kws_statesList=None, is_reverse=False, reduction="mean"):
        super(KWSIflytekLoss4, self).__init__()
        self.critirion_ce = nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction)
        self.node_none = node_none

        if kws_statesList != None:
            kws_statesList  = kws_statesList.split(',')
            kws_statesList  = Variable( torch.tensor(np.array( [int(x) for x in kws_statesList] )).reshape(-1)).detach()
        self.kws_statesList = kws_statesList
        self.is_reverse     = is_reverse

    def forward(self, student, teacher, frames_label, num_classes, nmod, diffCoefficent=0.0, sampleScore=1.0):
        '''
            x shape: N, 3003, T/nmod
            frames_label shape: N, T
        '''
        out_loss = torch.tensor(0.0).cuda() if student.is_cuda else torch.tensor(0.0)
        frames_label = frames_label.reshape(-1, 1)
        student = student.reshape(-1, student.size(-1))

        mask_kws = torch.ones(num_classes)
        if student.is_cuda:
            mask_kws = mask_kws.cuda()
            if self.kws_statesList != None:
                self.kws_statesList = self.kws_statesList.cuda()

        probs_student = F.softmax(student, dim=1)

        target = frames_label.clone().long().flatten()
        target = target.long().reshape(-1, nmod)[:, 0]        
        mask_ones   = torch.ones_like(target)
        mask_zeros  = torch.zeros_like(target)
        mask_ignore = torch.ones_like(target)*(-1)

        # Super
        target_super = target.clone()
        target_super[ target_super < 0 ] = -1
        if self.node_none != None:
            target_super[ target_super == self.node_none ] = -1
        gather = target_super.clone().view(-1, 1)
        gather[gather<0] = 0
        probs_student_super = torch.gather(probs_student, 1, gather).squeeze(1)
        probs_mask_super    = torch.where(target_super>0, mask_ones, mask_zeros)

        # # 难例采样
        # if sampleScore<1.0 and sampleScore>0.0:
        #     target_super = torch.where(probs_student_super<sampleScore, target_super, mask_ignore)

        # teache标签
        if teacher != None:
            teacher = teacher.reshape(-1, teacher.size(-1))
            assert(student.size()==teacher.size())
            if student.is_cuda:
                teacher = teacher.cuda()
        
            probs_teacher = F.softmax(teacher, dim=1)
        #     probs_teacher_super = torch.gather(probs_teacher, 1, gather).squeeze(1)
        
        #     if diffCoefficent > 0.0 and diffCoefficent < 1.0:
        #         probs_teacher_super = probs_teacher_super*(1-diffCoefficent) + diffCoefficent

        #     if self.kws_statesList != None:
        #         mask_kws.scatter_(0, self.kws_statesList, 0)
        #         target_kws = target_super.clone()
        #         target_kws[target_kws<0] = 3002
        #         states_kws = torch.gather(mask_kws, 0, target_kws)
        #         probs_teacher_super = torch.where(states_kws==0, probs_teacher_super, mask_ones.float())
            
        #     if self.is_reverse:
        #         probs_teacher_super = 1.0 - probs_teacher_super
        # else:
        probs_teacher_super = mask_ones.clone().float()

        # 计算loss
        log_super = (probs_student_super+1e-10).log()
        loss_super = -(probs_mask_super.detach() * probs_teacher_super.detach() )*log_super
        n_super = torch.sum(probs_mask_super)
        if n_super > 0:
            out_loss += loss_super.sum()/n_super

        # Unsuper
        if self.node_none != None and teacher != None:
            target_unsuper = target.clone()
            probs_teacher_unsuper, indices_teacher_unsuper = torch.max(probs_teacher, dim=1)
            probs_student_unsuper = torch.gather(probs_student, 1, indices_teacher_unsuper.view(-1, 1)).squeeze(1)

            probs_mask_unsuper = torch.where(target_unsuper==self.node_none, mask_ones, mask_zeros)

            log_unsuper = (probs_student_unsuper+1e-10).log()
            loss_unsuper = -(probs_mask_unsuper.detach() * probs_teacher_unsuper.detach())*log_unsuper

            n_unsuper = torch.sum(probs_mask_unsuper)
            if n_unsuper > 0:
                out_loss += loss_unsuper.sum()/n_unsuper
        return out_loss

class KWSIflytekLoss5(nn.Module):
    def __init__(self, node_none, reduction="mean"):
        super(KWSIflytekLoss5, self).__init__()
        self.critirion_ce  = nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction)
        self.critirion_kld = nn.KLDivLoss(reduction="none")
        self.node_none = node_none

    def forward(self, student, teacher, frames_label, num_classes, nmod):
        '''
            x shape: N, 3003, T/nmod
            frames_label shape: N, T
        '''
        out_loss = torch.tensor(0.0).cuda() if student.is_cuda else torch.tensor(0.0)
        frames_label = frames_label.reshape(-1, 1)
        student = student.reshape(-1, num_classes)

        target = frames_label.clone().long().flatten()
        target = target.reshape(-1, nmod)[:, 0] 

        # Super
        target_super = target.clone()
        target_super[ target_super < 0 ] = -1
        if self.node_none != None:
            target_super[ target_super == self.node_none ] = -1
        target_super = target_super.unsqueeze(-1)
        target_super = target_super.flatten().long()

        super_loss = self.critirion_ce(student, target_super)
        out_loss +=super_loss

        # teache标签
        if self.node_none != None and teacher != None:
            mask_ones  = torch.ones_like(target)
            mask_zeros = torch.zeros_like(target)
            teacher = teacher.reshape(-1, num_classes)
            assert(student.size()==teacher.size())
            if student.is_cuda:
                teacher = teacher.cuda()
            
            probs_teacher = F.softmax(teacher, dim=1)
            logit_student = F.log_softmax(student, dim=1)

            target_unsuper = target.clone()
            mask_unsuper   = torch.where(target_unsuper==self.node_none, mask_ones, mask_zeros)

            n_unsuper = torch.sum(mask_unsuper)
            if n_unsuper > 0:
                unsuper_loss = self.critirion_kld(logit_student, probs_teacher)
                unsuper_loss = torch.sum( (torch.sum(unsuper_loss, 1)*mask_unsuper.detach()) )/n_unsuper
                out_loss += unsuper_loss
        return out_loss

class KWSIflytekLoss6(nn.Module):
    def __init__(self, node_none):
        super(KWSIflytekLoss6, self).__init__()
        self.critirion_ce  = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self.critirion_kld = nn.KLDivLoss(reduction="none")
        self.node_none = node_none

    def forward(self, student, teacher, frames_label, num_classes, nmod):
        '''
            x shape: N, 3003, T/nmod
            frames_label shape: N, T
        '''
        out_loss = torch.tensor(0.0).cuda() if student.is_cuda else torch.tensor(0.0)
        frames_label = frames_label.reshape(-1, 1)
        student = student.reshape(-1, num_classes)

        target = frames_label.clone().long().flatten()
        target = target.reshape(-1, nmod)[:, 0] 

        # Super
        target_super = target.clone()
        target_super[ target_super < 0 ] = -1
        if self.node_none != None:
            target_super[ target_super == self.node_none ] = -1
        target_super = target_super.unsqueeze(-1)
        target_super = target_super.flatten().long()

        super_loss = self.critirion_ce(student, target_super)
        out_loss +=super_loss

        # Unsuper
        if self.node_none != None and teacher != None:
            mask_ones  = torch.ones_like(target)
            mask_zeros = torch.zeros_like(target)
            teacher = teacher.reshape(-1, num_classes)
            assert(student.size()==teacher.size())
            if student.is_cuda:
                teacher = teacher.cuda()

            probs_teacher = F.softmax(teacher, dim=1)
            probs_student = F.softmax(student, dim=1)

            target_unsuper = target.clone()
            probs_teacher_unsuper, indices_teacher_unsuper = torch.max(probs_teacher, dim=1)
            probs_student_unsuper = torch.gather(probs_student, 1, indices_teacher_unsuper.view(-1, 1)).squeeze(1)

            probs_mask_unsuper = torch.where(target_unsuper==self.node_none, mask_ones, mask_zeros)

            log_unsuper = (probs_student_unsuper+1e-10).log()
            loss_unsuper = -(probs_mask_unsuper.detach() * probs_teacher_unsuper.detach())*log_unsuper

            n_unsuper = torch.sum(probs_mask_unsuper)
            if n_unsuper > 0:
                out_loss += loss_unsuper.sum()/n_unsuper

        return out_loss

class KWSIflytekLoss7(nn.Module):
    def __init__(self, node_none, kws_phoneList, weight=None, label_smoothing=0.0, is_kld=False, kld_gamma=0.5):
        super(KWSIflytekLoss7, self).__init__()
        self.critirion_ce  = nn.CrossEntropyLoss(weight=weight, ignore_index=-1, reduction="mean", label_smoothing=label_smoothing)
        self.critirion_kld = nn.KLDivLoss(reduction='none')
        self.node_none     = node_none
        self.is_kld        = is_kld
        self.kld_gamma     = kld_gamma

        self.kws_phoneList = kws_phoneList
        if self.kws_phoneList != None and kws_phoneList != None:
            self.kws_phoneList = self.kws_phoneList.split(',')
            self.kws_phoneList = Variable( torch.tensor(np.array( [int(x) for x in self.kws_phoneList] )).reshape(1, -1) ).detach()

    def forward(self, student, teacher, frames_label, num_classes, nmod, frames_e2e, langid):
        '''
            x shape: N, 3003, T/nmod
            frames_label shape: N, T
        '''
        out_loss = torch.tensor(0.0).cuda() if student.is_cuda else torch.tensor(0.0)
        frames_label = frames_label.reshape(-1, 1)
        frames_e2e = frames_e2e.reshape(-1, 1)
        student = student.reshape(-1, num_classes)
        
        if teacher != None:
            teacher = teacher.reshape(-1, num_classes)
            assert(student.size()==teacher.size())
            if student.is_cuda:
                teacher = teacher.cuda()
        
        target = frames_label.clone().long().flatten()
        target_e2e = frames_e2e.clone().long().flatten()
        target = target.reshape(-1, nmod)[:, 0] 
        target_e2e = target_e2e.reshape(-1, nmod)[:, 0] 

        #对于kld训练来说，非当前语种硬标签可置为0，teacher的软标签不可置为0，需为-1，否则训练目标变为实际状态0
        # langid需为另一个语种的id，非当前语种
        lang_mask = target_e2e.clone()
        lang_mask = torch.where(lang_mask == langid, torch.zeros_like(lang_mask), torch.full_like(lang_mask, 1))


        mask_ones  = torch.ones_like(target)
        mask_zeros = torch.zeros_like(target)

        # Super
        target_super = target.clone()
        target_super[ target_super < 0 ] = -1
        if self.node_none != None:
            target_super[ target_super == self.node_none ] = -1
        mask_super = torch.where(target_super>=0, mask_ones, mask_zeros)
        super_loss, loss_unsuper = torch.tensor(0.0).cuda(),torch.tensor(0.0).cuda()
        mask_super = mask_super*lang_mask
        n_super    = torch.sum(mask_super)
        if self.is_kld and teacher != None:
            probs_teacher = F.softmax(teacher, dim=1).detach()
            
            logit_student = F.log_softmax(student, dim=1)
            soft_target   = self.kld_gamma*F.one_hot((target_super*mask_super).long(), num_classes=num_classes) + (1-self.kld_gamma)*probs_teacher
            super_loss    = self.critirion_kld(logit_student, soft_target)
            if n_super > 0:
                super_loss = torch.sum((torch.sum(super_loss, dim=1)*mask_super))/float(n_super)
                out_loss += super_loss
        else:
            target_super = target_super.unsqueeze(-1)
            target_super = target_super.flatten().long()
            if n_super > 0:
                super_loss = self.critirion_ce(student, target_super)
                out_loss += super_loss

        # Unsuper
        if self.node_none != None and self.kws_phoneList != None:
            if student.is_cuda: 
                self.kws_phoneList = self.kws_phoneList.cuda()

            target_unsuper = target.clone()
            mask_unsuper   = torch.where(target_unsuper==self.node_none, mask_ones, mask_zeros)
            n_unsuper      = torch.sum(mask_unsuper)
            
            N, C = student.size()
            states_mask = student.data.new(N, C).fill_(1)
            states_mask = Variable(states_mask)
            _, ids_sel  = torch.broadcast_tensors(student[:, :self.kws_phoneList.size(1)], self.kws_phoneList)
            states_mask.scatter_(1, ids_sel.data, 0)

            probs_student      = F.softmax(student, dim=1)
            probs_student_mask = probs_student * states_mask.detach()
            probs_student_mask = torch.sum(probs_student_mask, dim=1)

            if teacher != None:
                probs_teacher      = F.softmax(teacher, dim=1).detach()
                probs_teacher_mask = probs_teacher * states_mask.detach()
                probs_teacher_mask = torch.sum(probs_teacher_mask, dim=1)
                loss_unsuper = -(probs_teacher_mask*0.5 + 0.5) * (probs_student_mask.log())
            else:
                loss_unsuper = - probs_student_mask.log()

            if n_unsuper > 0:
                loss_unsuper = torch.sum(loss_unsuper*mask_unsuper)/float(n_unsuper)
                out_loss += loss_unsuper
        return super_loss, loss_unsuper 

class FocalLoss(nn.Module):
    """
        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    """
    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x, label, numclasses, nmod):
        target = label.clone().reshape(-1, 1)
        target = target.flatten()
        target[target<0] = -1
        target[target>=numclasses] = -1
        target = target.long()
        target = target.reshape(-1, nmod)[:,0].unsqueeze(0)

        # mask
        mask_ones  = torch.ones_like(target)
        mask_zeros = torch.zeros_like(target)
        mask_focal = torch.where(target>=0, mask_ones, mask_zeros) 
        mask_focal = mask_focal.reshape(-1, 1)

        # gather prob
        x = x.reshape(-1, x.size(-1))
        gather = target.clone().view(-1, 1)
        gather[gather<0] = 0
        probs = F.softmax(x, dim=1)
        probs = torch.gather(probs, 1, gather).view(-1, 1)
        log_p = (probs+1e-10).log()

        loss_focal = -mask_focal*(torch.pow((1-probs), self.gamma))*log_p

        loss_out = torch.tensor(0.0)
        if self.reduction=="mean" and torch.sum(mask_focal) > 0:
            loss_out = loss_focal.sum()/torch.sum(mask_focal)
        else:
            loss_out = loss_focal
        return loss_out

class MseLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MseLoss, self).__init__()
        self.mseloss = nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        x1 = x.reshape(-1, x.size(-1))
        y1 = y.clone().reshape(-1, y.size(-1)).detach()
        mse_loss = self.mseloss(x1, y1)
        return mse_loss

class KldLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(KldLoss, self).__init__()
        self.kld_critirion   = nn.KLDivLoss(reduction=reduction)

    def forward(self, x, y):
        x1 = x.clone().reshape(-1, x.size(-1))
        x1 = F.log_softmax(x1, dim=1)
        y1 = y.clone().reshape(-1, y.size(-1))
        y1 = F.softmax(y1, dim=1).detach()
        kld_loss = self.kld_critirion(x1, y1)
        return kld_loss

class KldLossWithSample(nn.Module):
    def __init__(self, unitList_sample, train_node_None, reduction='none'):
        super(KldLossWithSample, self).__init__()
        self.kld_critirion   = nn.KLDivLoss(reduction=reduction)
        self.unitList_sample = Variable( torch.tensor(np.array( [int(x) for x in unitList_sample.split(',')] )).reshape(-1) )
        self.train_node_None = train_node_None

    def func_hardSample(self, x, label, sampleScore):
        gather = label.clone().long()
        gather[gather<0] = 0

        probs = x.clone()
        probs = F.softmax(probs, dim=1)
        probs = torch.gather(probs, 1, gather.unsqueeze(1)).squeeze(1)

        if label.is_cuda and not self.unitList_sample.is_cuda:
            self.unitList_sample = self.unitList_sample.cuda()
        mask_sample = torch.ones_like(label)
        for index, value in enumerate(label):
            if value in self.unitList_sample and probs[index] > sampleScore:
                mask_sample[index] = 0.0
        return mask_sample

    def forward(self, x, y, frames_label, nmod, sampleScore):
        x1  = x.clone().reshape(-1, x.size(-1))
        target = frames_label.clone().flatten()
        target[ target < 0 ] = -1
        target[ target == self.train_node_None ] = -1
        target = target.long().reshape(-1, nmod)[:, 0]
        mask_sample = self.func_hardSample(x1, target, sampleScore)
        
        x1 = F.log_softmax(x1, dim=1)
        y1 = y.clone().reshape(-1, y.size(-1))
        y1 = F.softmax(y1, dim=1).detach()
        kld_loss = self.kld_critirion(x1, y1)
        
        kld_loss  = torch.sum((torch.sum(kld_loss, 1)*mask_sample))/torch.sum(mask_sample)
        return kld_loss

class CeLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CeLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction)

    def func_hardSample(self, x, label, sampleScore):
        gather = label.clone().long()
        gather[gather<0] = 0

        probs = x.clone()
        probs = F.softmax(probs, dim=1)
        probs = torch.gather(probs, 1, gather.unsqueeze(1)).squeeze(1)

        # hardSample标签
        mask = probs.clone()
        matrix_neg = torch.ones_like(mask) * (-1)
        label_hardSample = torch.where(mask<sampleScore, label, matrix_neg.long())

        return label_hardSample

    def forward(self, x, frames_label, num_classes, nmod, sampleScore=1.0):
        
        frames_label = frames_label.reshape(-1, 1)
        x = x.reshape(-1, x.size(-1))
        
        target = frames_label.clone()
        target = target.flatten()
        target[ target<0 ] = -1
        target[ target>=num_classes ] = -1
        target = target.long()
        target = target.view(-1, nmod)[:,0]
        target = target.unsqueeze(-1)
        target = target.flatten().long()

        if sampleScore < 1.0 and sampleScore > 0.0:
            target = self.func_hardSample(x, target, sampleScore)
        
        x = x.reshape((-1, num_classes))
        
        ce_loss = self.celoss(x, target)
        return ce_loss

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, x, target, nmod):
        target = target.clone()
        target[ target<0 ] = -1
        target = target.view(-1, nmod)[:,0]
        target = target.unsqueeze(-1)
        target = target.flatten().float()

        remove_pad_mask = target.ne(-1)
        x = x[remove_pad_mask]
        target = target[target != -1]
        
        loss = self.bceloss(x, target)
        return loss

class FocalLossWithWeight(nn.Module):
    """
        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    """
    def __init__(self, numclasses, alpha=None, gamma=2, size_average=True):
        super(FocalLossWithWeight, self).__init__()
        if alpha is None:
            alpha = torch.ones(numclasses+1)
            alpha[-1:] = 0.0
            self.alpha = Variable(alpha)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.numclasses = numclasses
        self.size_average = size_average

    def forward(self, inputs, targets, nmod):
        targets = targets.reshape(-1, 1)
        inputs = inputs.reshape(-1, inputs.size(-1))
        
        targets = targets.flatten()
        targets[targets<0] = -1
        targets[targets>=self.numclasses] = -1
        targets = targets.long()
        targets = targets.reshape(-1, nmod)[:,0].unsqueeze(0)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        # mask weight
        targets_weight = targets.clone()
        targets_weight[targets_weight<0] = inputs.shape[1]
        ids = targets_weight.view(-1, 1)
        w_alpha = self.alpha[ids.data.view(-1)].view(-1, 1)

        # gather prob   
        class_gather = targets.clone().view(-1, 1)
        class_gather[class_gather<0] = 0
        probs = F.softmax(inputs, dim=1)
        probs = torch.gather(probs, 1, class_gather).view(-1, 1)
        log_p = (probs+1e-10).log()

        batch_loss = -w_alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            w_alpha [ w_alpha>0 ] = 1
            if torch.sum(w_alpha) > 0:
                loss = batch_loss.sum()/torch.sum(w_alpha)
            else:
                loss = batch_loss.sum()
        else:
            loss = batch_loss.sum()
        return loss

class KLDLossWithLable(nn.Module):
    def __init__(self, kld_gamma):
        super(KLDLossWithLable, self).__init__()
        self.kld_gamma     = kld_gamma
        self.kld_critirion = nn.KLDivLoss(reduction='none')

    def forward(self, logit_s, logit_t, frames_label, numclasses, nmod):
        '''
            logit_s shape: N, 3003, T/nmod
            logit_t shape: N, 3003, T/nmod
            frames_label shape: N, T
        '''
        target = frames_label.clone()
        target = target.flatten()
        target[target<0] = -1
        target[target>=numclasses] = -1
        target = target.long()
        target = target.reshape(-1, nmod)[:,0]
        
        mask_ones = torch.ones_like(target)
        mask_zeros = torch.zeros_like(target)
        kld_mask = torch.where(target>=0, mask_ones, mask_zeros)

        logit_t = F.softmax(logit_t, 1).detach()
        logit_s = F.log_softmax(logit_s, 1)

        kld_tar = (1-self.kld_gamma)*F.one_hot((target*kld_mask).long(), num_classes=numclasses) + self.kld_gamma*logit_t
        kld_loss = self.kld_critirion(logit_s, kld_tar)
        kld_loss = torch.sum((torch.sum(kld_loss, 1)*kld_mask))/torch.sum(kld_mask)

        return kld_loss

class CeLoss_smooth(nn.Module):
    def __init__(self, smooth=0.95):
        super(CeLoss_smooth, self).__init__()
        self.smooth = smooth

    def forward(self, x: torch.Tensor, att_label: torch.Tensor) -> torch.Tensor:
        lprob, target = self.getlabel(x, att_label)
        smooth_loss = self.get_smooth_loss(lprob, target)
        return smooth_loss

    def getlabel(self, x, att_label):
        lprob = nn.functional.log_softmax(x, dim=1)
        num_classes = x.shape[1]
        ori_att_label = att_label.clone()
        ori_att_label = ori_att_label.flatten()
        ori_att_label[ori_att_label < 0] = -1
        ori_att_label = ori_att_label.long()
        target = ori_att_label
        target = target.unsqueeze(-1)
        _, new_target = torch.broadcast_tensors(lprob, target)
        remove_pad_mask = new_target.ne(-1)
        lprob = lprob[remove_pad_mask]
        lprob = lprob.reshape((-1, num_classes))
        target = target[target != -1]
        target = target.unsqueeze(-1)
        return lprob, target

    def get_smooth_loss(self, lprob, target):
        valid = lprob.shape[0]
        num_classes = lprob.shape[1]
        ####################    
        smooth_label = torch.ones_like(lprob) * (1 - self.smooth) / (num_classes - 1)
        smooth_label[range(smooth_label.shape[0]), target.reshape(-1, ).long()] = self.smooth
        smooth_loss = -(smooth_label * lprob).sum() / valid
        return smooth_loss

class CTCLoss(nn.Module):
    def __init__(self, blankID):
        super(CTCLoss, self).__init__()
        self.ctcLoss = nn.CTCLoss(blank=blankID, reduction='mean', zero_infinity=False)

    def forward(self, x, ctc_list, x_len , ctc_len):
        log_p = F.log_softmax(x, dim=2)
        ctc_loss = self.ctcLoss(log_p, ctc_list, x_len, ctc_len)
        return ctc_loss

class SI_SNR(nn.Module):
    def __init__(self, loss_weight):
        super(SI_SNR, self).__init__()
        self.loss_weight = loss_weight
        self.eps = 1e-8

    def forward(self, output, target):
        max_values, _ = torch.max(torch.abs(target), dim=1)
        max_values=max_values.unsqueeze(1).repeat(1, target.size(1))+1e-10
        output = output - torch.mean(output, dim=1, keepdim=True)
        target = target - torch.mean(target, dim=1, keepdim=True)
        output = output * self.loss_weight / max_values / 2
        target = target * self.loss_weight / max_values / 2
        e_noise = output - target
        si_snr = -10 * torch.log10( torch.sum( target**2, dim=1, keepdim=True) / (torch.sum( e_noise**2, dim=1, keepdim=True) + self.eps ) + self.eps )
        si_snr = torch.mean( si_snr )
        return si_snr

class SI_SDR(nn.Module):
    def __init__(self, loss_norm):
        super(SI_SDR, self).__init__()
        self.loss_norm = loss_norm
        self.eps = 1e-8

    def remove_dc(self, signal):
        """Normalized to zero mean"""
        mean = torch.mean(signal, dim=-1, keepdim=True)
        signal = signal - mean
        return signal

    def pow_p_norm(self, signal):
        """Compute 2 Norm"""
        return torch.pow(torch.norm(signal, p=2, dim=-1, keepdim=True), 2)

    def pow_norm(self, s1, s2):
        return torch.sum(s1 * s2, dim=-1, keepdim=True)

    def forward(self, estimated, original):
        if self.loss_norm:
            estimated = self.remove_dc(estimated)
            original = self.remove_dc(original)
        target = self.pow_norm(estimated, original) * original / (self.pow_p_norm(original) + self.eps)
        noise = estimated - target
        si_sdr = 10 * torch.log10(self.pow_p_norm(target) / (self.pow_p_norm(noise) + self.eps) + self.eps)
        si_sdr = torch.mean( si_sdr)
        return si_sdr
