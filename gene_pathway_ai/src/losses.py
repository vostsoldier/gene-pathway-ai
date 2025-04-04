import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, margin: float = 1.0,
                 weight_focal: float = 0.7, weight_contrastive: float = 0.3):
        """
        alpha: balancing factor for positive class in focal loss
        gamma: focusing parameter in focal loss
        margin: distance margin for contrastive loss
        weight_focal: weight for focal loss component
        weight_contrastive: weight for contrastive loss component
        """
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.margin = margin
        self.w_focal = weight_focal
        self.w_contrastive = weight_contrastive

    def focal_loss(self, logits, targets):
        """
        Computes focal loss for binary classification.
        logits: raw model predictions (before sigmoid), shape [batch, 1]
        targets: ground-truth labels (0/1), shape [batch, 1]
        """
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        # Weight positive examples by alpha and negatives by (1-alpha)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_factor * focal_term * ce_loss
        return loss.mean()

    def contrastive_loss(self, features, targets):
        """
        Computes contrastive loss on latent features.
        features: fused latent features with shape [batch, d]
        targets: ground-truth labels with shape [batch, 1] or [batch]
        For pairs with the same target, loss = d^2.
        For pairs with different targets, loss = max(0, margin - d)^2
        """
        if targets.dim() > 1:
            targets = targets.view(-1)
        # Compute pairwise Euclidean distances
        distances = torch.cdist(features, features, p=2)  # [batch, batch]
        targets = targets.unsqueeze(1)  # [batch, 1]
        mask_same = (targets == targets.t()).float()
        mask_diff = 1 - mask_same
        loss_same = mask_same * (distances ** 2)
        loss_diff = mask_diff * torch.clamp(self.margin - distances, min=0) ** 2
        # Average over all pairs (symmetric)
        loss = (loss_same + loss_diff).mean()
        return loss

    def forward(self, logits, targets, latent_features):
        lf = self.focal_loss(logits, targets)
        lc = self.contrastive_loss(latent_features, targets)
        return self.w_focal * lf + self.w_contrastive * lc