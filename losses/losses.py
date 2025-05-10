import torch
import torch.nn as nn
import torch.nn.functional as F

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# Dice Loss with Class Weights
def dice_loss(pred, target, class_weights=None, epsilon=1e-6):
    pred_soft = F.softmax(pred, dim=1)
    B, C, H, W = pred_soft.shape
    target_onehot = torch.zeros(B, C, H, W, device=pred.device)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

    dims = (0, 2, 3)
    intersection = torch.sum(pred_soft * target_onehot, dims)
    cardinality = torch.sum(pred_soft + target_onehot, dims)
    dice_per_class = (2. * intersection + epsilon) / (cardinality + epsilon)

    if class_weights is not None:
        dice_loss = 1.0 - (dice_per_class * class_weights).sum() / class_weights.sum()
    else:
        dice_loss = 1.0 - dice_per_class.mean()
    return dice_loss

# Main Loss Class
class compute_losses(nn.Module):
    def __init__(self, device='cuda', loss_mode='ce+focal+dice'):
        super(compute_losses, self).__init__()
        self.device = device
        self.loss_mode = loss_mode  # Options: 'ce', 'ce+dice', 'focal+dice', 'ce+focal+dice'
        self.L1Loss = nn.L1Loss()

        # Class weights (adjust if needed)
        self.class_weights = torch.tensor([0.004, 0.02, 5.0, 10.0]).to(device)

        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.focal_loss = FocalLoss(alpha=self.class_weights, gamma=2.0)
        self.dice_weights = self.class_weights

    def forward(self, opt, weight, inputs, outputs, features, retransform_features):
        losses = {}
        type = opt.type

        losses["topview_loss"] = self.compute_topview_loss(outputs["topview"], inputs[type])
        losses["transform_topview_loss"] = self.compute_topview_loss(outputs["transform_topview"], inputs[type])
        losses["transform_loss"] = self.compute_transform_loss(features, retransform_features)

        losses["loss"] = (
            losses["topview_loss"] +
            0.001 * losses["transform_loss"] +
            losses["transform_topview_loss"]
        )

        return losses

    def compute_topview_loss(self, pred, target):
        target = torch.squeeze(target.long())

        if self.loss_mode == 'ce':
            return self.ce_loss(pred, target)

        elif self.loss_mode == 'ce+dice':
            ce = self.ce_loss(pred, target)
            dice = dice_loss(pred, target, class_weights=self.dice_weights)
            return 0.7 * ce + 0.3 * dice

        elif self.loss_mode == 'focal+dice':
            focal = self.focal_loss(pred, target)
            dice = dice_loss(pred, target, class_weights=self.dice_weights)
            return 0.7 * focal + 0.3 * dice

        elif self.loss_mode == 'ce+focal+dice':
            ce = self.ce_loss(pred, target)
            focal = self.focal_loss(pred, target)
            dice = dice_loss(pred, target, class_weights=self.dice_weights)
            return (0.4 * ce + 0.3 * focal + 0.3 * dice)

        else:
            raise ValueError(f"Invalid loss mode: {self.loss_mode}")

    def compute_transform_loss(self, outputs, retransform_output):
        return self.L1Loss(outputs, retransform_output)


