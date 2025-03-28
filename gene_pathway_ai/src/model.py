import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from torch.utils.checkpoint import checkpoint

class GeneEncoder(nn.Module):
    def __init__(self, seq_len=10000, embed_dim=64):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=9, padding=4), 
            nn.ReLU(),
            nn.MaxPool1d(2),  
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(5), 
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(10),
        )
        self.adaptive_pool = nn.AdaptiveMaxPool1d(100)
        self.fc = nn.Linear(64 * 100, embed_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class PathwayEncoder(nn.Module):
    def __init__(self, in_feats: int = 4, hidden_dim: int = 64, embed_dim: int = 64):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_dim // 2, heads=2)
        self.conv2 = GATConv(hidden_dim, embed_dim, heads=1, concat=False)
        self.dropout = nn.Dropout(0.2)
        self.edge_transform = nn.Linear(2, hidden_dim)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        
        return x

class FusionModel(nn.Module):
    def __init__(self, seq_len: int = 10000, pathway_feat_dim: int = 4, 
                 embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        
        self.gene_enc = GeneEncoder(seq_len=seq_len, embed_dim=embed_dim)
        self.pathway_enc = PathwayEncoder(in_feats=pathway_feat_dim, 
                                          hidden_dim=hidden_dim, 
                                          embed_dim=embed_dim)
        fusion_input_dim = embed_dim * 2 
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, genes: torch.Tensor, pathway: torch.Tensor) -> torch.Tensor:
        gene_embed = self.gene_enc(genes)
        path_embed = self.pathway_enc(pathway)
        if gene_embed.size(0) > path_embed.size(0):
            path_embed = path_embed.repeat(gene_embed.size(0), 1)
        combined = torch.cat([gene_embed, path_embed], dim=1)
        out = self.fusion_layers(combined)
        
        return out