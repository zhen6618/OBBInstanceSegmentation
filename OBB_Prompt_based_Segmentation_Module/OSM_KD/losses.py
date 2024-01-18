import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, inputs, targets):
        # inputs = F.sigmoid(inputs)
        # targets = F.sigmoid(targets)

        inputs = inputs.view(-1, 1)
        targets = targets.view(-1, 1)

        inputs = torch.concat([inputs, (1 - inputs)], dim=1)
        targets = torch.concat([targets, (1 - targets)], dim=1)

        loss = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return loss


class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, inputs, targets):

        loss = F.mse_loss(inputs, targets, reduction='mean')

        return loss


class KLDLoss(nn.Module):

    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, input_probs, target_probs):
        assert input_probs.shape == target_probs.shape

        input_probs = F.sigmoid(input_probs)
        target_probs = F.sigmoid(target_probs)

        input_probs = input_probs.view(-1, 1)
        target_probs = target_probs.view(-1, 1)

        input_probs = torch.concat([input_probs, (1 - input_probs)], dim=1)
        target_probs = torch.concat([target_probs, (1 - target_probs)], dim=1)

        kld_loss = F.kl_div(input_probs.log(), target_probs, reduction='batchmean')

        return kld_loss


def calc_iou_t(student_pred_mask: torch.Tensor, teacher_pred_mask: torch.Tensor):
    student_pred_mask = F.sigmoid(student_pred_mask)
    student_pred_mask = torch.clamp(student_pred_mask, min=0, max=1)

    teacher_pred_mask = F.sigmoid(teacher_pred_mask)
    teacher_pred_mask = torch.clamp(teacher_pred_mask, min=0, max=1)

    student_pred_mask = (student_pred_mask >= 0.5).float()
    teacher_pred_mask = (teacher_pred_mask >= 0.5).float()

    intersection = torch.sum(torch.mul(student_pred_mask, teacher_pred_mask), dim=(1, 2))
    union = torch.sum(student_pred_mask, dim=(1, 2)) + torch.sum(teacher_pred_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou

