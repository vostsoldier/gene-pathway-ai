from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data.data import Data, DataTensorAttr, EdgeAttr
from torch.serialization import add_safe_globals
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.graph_store import EdgeLayout
from torch_geometric.data.storage import GlobalStorage, EdgeStorage, NodeStorage
import os
add_safe_globals([
    Data, 
    DataTensorAttr, 
    EdgeAttr, 
    DataEdgeAttr, 
    EdgeLayout,
    GlobalStorage, 
    EdgeStorage, 
    NodeStorage
])


class DNABERTEncoder(nn.Module):
    def __init__(self, pretrained_model="zhihan1996/DNABERT-2-117M", output_dim=256, 
                 use_gradient_checkpointing=False, seq_len=512):
        super().__init__()
        self.seq_len = seq_len
        self.dnabert = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True)
        hidden_dim = self.dnabert.config.hidden_size
        self.projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()
        
        batch_size = x.size(0)
        all_embeddings = []
        
        for i in range(batch_size):
            single_seq = x[i:i+1]
            mask = (single_seq != 0).float()
            
            try:
                outputs=self.dnabert.embeddings(single_seq)
                hidden_states = self.dnabert.encoder.layer[0].attention.self.key(outputs)
                embedding = outputs[0][:, 0].squeeze(0)
            except Exception as e:
                with torch.set_grad_enabled(True):
                    embedding = self.dnabert.embeddings.word_embeddings(single_seq).mean(dim=1).squeeze(0)
            all_embeddings.append(embedding)
        embeddings = torch.stack(all_embeddings)
        return self.projection(embeddings)


class PathwayGNN(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=16, out_channels=256, edge_dim=2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=2, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_channels * 2, out_channels, heads=1, edge_dim=edge_dim)
        
    def forward(self, data, return_node_features=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        if return_node_features:
            return x
        x = global_mean_pool(x, data.batch if hasattr(data, 'batch') else None)
        return x

class CrossModalAttentionLayer(nn.Module):
    def __init__(self, gene_dim, pathway_dim, attn_dim):
        super().__init__()
        self.query_proj = nn.Linear(gene_dim, attn_dim)
        self.key_proj = nn.Linear(pathway_dim, attn_dim)
        self.value_proj = nn.Linear(pathway_dim, attn_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, gene_embedding, pathway_node_features):
        batch_size = gene_embedding.size(0)
        pathway_nodes = pathway_node_features.unsqueeze(0).expand(batch_size, -1, -1)
        Q = self.query_proj(gene_embedding).unsqueeze(1)
        K = self.key_proj(pathway_nodes)
        V = self.value_proj(pathway_nodes)
        scores = torch.matmul(Q, K.transpose(1, 2))
        attn_weights = self.softmax(scores)
        attn_output = torch.matmul(attn_weights, V)
        return attn_output.squeeze(1), attn_weights.squeeze(1)

class FusionModel(nn.Module):
    def __init__(self, gene_dim: int = 256, pathway_dim: int = 256, 
                 hidden_dims: list = [128, 64], output_dim: int = 1,
                 use_gradient_checkpointing: bool = False, seq_len: int = 512,
                 embed_dim: Optional[int] = None, hidden_dim: Optional[int] = None,
                 pathway_data: Optional[object] = None, use_disease_data: bool = False,
                 disease_emb_dim: int = 16):
        super().__init__()
        if embed_dim is not None:
            gene_dim = embed_dim
        if hidden_dim is not None:
            hidden_dims = [hidden_dim, hidden_dim // 2]
        self.use_disease_data = use_disease_data
        self.gene_enc = DNABERTEncoder(
            output_dim=gene_dim, 
            use_gradient_checkpointing=use_gradient_checkpointing,
            seq_len=seq_len
        )
        pathway_in_channels = 4  
        edge_dim = 2
        if pathway_data is not None:
            if hasattr(pathway_data, 'x'):
                pathway_in_channels = pathway_data.x.size(1)
            if hasattr(pathway_data, 'edge_attr'):
                edge_dim = pathway_data.edge_attr.size(1)
        print("Using pathway_in_channels =", pathway_in_channels, ", edge_dim =", edge_dim)
        self.pathway_enc = PathwayGNN(
            in_channels=pathway_in_channels, 
            edge_dim=edge_dim, 
            out_channels=pathway_dim
        )
        self.cross_attn = CrossModalAttentionLayer(gene_dim, pathway_dim, attn_dim=gene_dim)
        
        if self.use_disease_data:
            self.disease_encoder = nn.Sequential(
                nn.Linear(1, disease_emb_dim),
                nn.ReLU()
            )
            mlp_input_dim = gene_dim + gene_dim + disease_emb_dim
        else:
            mlp_input_dim = gene_dim + gene_dim
        layers = []
        dims = [mlp_input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, gene_seq: torch.Tensor, pathway_data, gene_names: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gene_embedding = self.gene_enc(gene_seq)
        if gene_embedding.dim() > 2:
            gene_embedding = gene_embedding.mean(dim=1)
        pathway_nodes = self.pathway_enc(pathway_data, return_node_features=True)
        cross_output, attn_weights = self.cross_attn(gene_embedding, pathway_nodes)
        if cross_output.dim() > 2:
            cross_output = cross_output.mean(dim=1)
        
        if self.use_disease_data and gene_names is not None:
            from gene_pathway_ai.src.ensembl_api import get_gene_disease_associations
            disease_counts = []
            for gene in gene_names:
                data = get_gene_disease_associations(gene)
                count = len(data) if (data is not None and isinstance(data, list)) else 0
                disease_counts.append([count])
            disease_tensor = torch.tensor(disease_counts, dtype=torch.float, device=gene_embedding.device)
            disease_embedding = self.disease_encoder(disease_tensor)
            combined = torch.cat([gene_embedding, cross_output, disease_embedding], dim=1)
        else:
            combined = torch.cat([gene_embedding, cross_output], dim=1)
        
        output = self.mlp(combined)
        return output, attn_weights, combined


def tokenize_dna_sequences(sequences: list, max_length: int = 512) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
    encoded_inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    )
    
    return encoded_inputs.input_ids
'''
model = FusionModel(use_gradient_checkpointing=False)
dna_sequences = ["ATGCATGCATGC", "GATTACAGATCG"] 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  
graph_path = os.path.join(parent_dir, "data", "dna_repair_graph.pt")
pathway_data = torch.load(graph_path)
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
tokens = tokenizer(dna_sequences, return_tensors="pt", padding=True)
predictions = model(tokens.input_ids, pathway_data)
'''