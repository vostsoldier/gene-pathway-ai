import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):  
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets, attention_weights=None):
        bce_loss = self.bce(predictions, targets)
        pt = torch.exp(-bce_loss)
        focal_weight = (1-pt)**2
        focal_loss = focal_weight * bce_loss
        
        total_loss = self.alpha * focal_loss
        
        if attention_weights is not None:
            attn_diversity_loss = -torch.mean(
                torch.std(attention_weights, dim=1)
            )
            total_loss += self.beta * attn_diversity_loss
        
        return total_loss