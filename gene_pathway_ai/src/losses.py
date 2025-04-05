import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 weight_focal: float = 0.7, weight_contrastive: float = 0.3):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.w_focal = weight_focal
        self.w_contrastive = weight_contrastive
        
    def focal_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_factor * focal_term * ce_loss
        return loss.mean()
    
    def attention_contrastive_loss(self, attn_weights, targets):
        if targets.dim() > 1:
            targets = targets.view(-1)
        
        pos_mask = (targets == 1).float()
        node_attention_avg = attn_weights.mean(dim=1)  
        loss = pos_mask * (1 - node_attention_avg) + (1 - pos_mask) * node_attention_avg
        
        return loss.mean()
    
    def forward(self, logits, targets, attn_weights):
        lf = self.focal_loss(logits, targets)
        lc = self.attention_contrastive_loss(attn_weights, targets)
        return self.w_focal * lf + self.w_contrastive * lc