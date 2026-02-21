import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        :param alpha: 缩放因子，平衡正负样本的比例
        :param gamma: 调节因子，控制易分类样本和难分类样本的损失贡献
        :param reduction: 指定损失的计算方式，'mean' 或 'sum' 或 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: 模型的预测结果 (logits)，形状为 (batch_size, 1)
        :param targets: 实际标签，形状为 (batch_size, 1)
        :return: 计算得到的 Focal Loss
        """
        # 将 targets 转换为浮点型，确保与 inputs 类型一致
        targets = targets.type_as(inputs)

        # 计算二元交叉熵损失（带 logits 的版本）
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 计算 p_t
        pt = torch.exp(-BCE_loss)  # p_t = exp(-BCE)

        # 计算 Focal Loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class FocalLossweight(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        :param alpha: 可设置为一个值或一个包含 [alpha_positive, alpha_negative] 的列表，平衡正负样本的比例
        :param gamma: 调节因子，控制易分类样本和难分类样本的损失贡献
        :param reduction: 指定损失的计算方式，'mean' 或 'sum' 或 'none'
        """
        super(FocalLossweight, self).__init__()
        if alpha is None:
            self.alpha = torch.tensor([0.25, 0.75])  # 默认正负样本比例为1:3时的alpha值
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: 模型的预测结果 (logits)，形状为 (batch_size, 1)
        :param targets: 实际标签，形状为 (batch_size, 1)
        :return: 计算得到的 Focal Loss
        """
        targets = targets.type_as(inputs)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha[targets.long().view(-1)].cuda()

        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

