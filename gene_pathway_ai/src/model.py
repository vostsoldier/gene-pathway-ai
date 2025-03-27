import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.checkpoint import checkpoint

class GeneEncoder(nn.Module):
    def __init__(self, seq_len: int = 10000, embed_dim: int = 32):
        super().__init__()
        self.conv = nn.Conv1d(5, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear((seq_len // 2) * 8, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = checkpoint(self.conv, x)  
        x = checkpoint(self.pool, F.relu(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class PathwayEncoder(nn.Module):
    def __init__(self, in_feats: int = 3, embed_dim: int = 32):
        super().__init__()
        self.conv1 = GCNConv(in_feats, 16)
        self.conv2 = GCNConv(16, embed_dim)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = checkpoint(self.conv1, x, edge_index)
        x = F.relu(x)
        x = checkpoint(self.conv2, x, edge_index)
        x = global_mean_pool(x, batch)
        return x

class FusionModel(nn.Module):
    def __init__(self, seq_len: int = 10000):
        super().__init__()
        self.gene_enc = GeneEncoder(seq_len=seq_len)
        self.pathway_enc = PathwayEncoder()
        self.dropout = nn.Dropout(p=0.3)
        self.mlp = nn.Sequential(
            nn.Linear(32 + 32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(16, 1)
        )

    def forward(self, gene_seq: torch.Tensor, pathway_data) -> torch.Tensor:
        gene_embed = self.gene_enc(gene_seq)
        path_embed = self.pathway_enc(pathway_data)
        path_embed = path_embed.repeat(gene_embed.size(0), 1)
        fused = torch.cat([gene_embed, path_embed], dim=1)
        fused = self.dropout(fused)  
        return self.mlp(fused)