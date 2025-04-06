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
import math
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
        self.hidden_channels = hidden_channels
        self.edge_types = ['activation', 'inhibition', 'undefined']
        self.conv1_dict = nn.ModuleDict({
            'activation': GATConv(in_channels, hidden_channels, heads=2, edge_dim=edge_dim-1),
            'inhibition': GATConv(in_channels, hidden_channels, heads=2, edge_dim=edge_dim-1),
            'undefined': GATConv(in_channels, hidden_channels, heads=2, edge_dim=edge_dim-1)
        })
        
        self.conv2 = GATConv(hidden_channels * 2, out_channels, heads=1, edge_dim=edge_dim)
        
    def forward(self, data, return_node_features=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_type_outputs = []
        activation_mask = (edge_attr[:,0] > 0.5) 
        inhibition_mask = (edge_attr[:,0] < -0.5) 
        undefined_mask = (~activation_mask & ~inhibition_mask)  
        
        type_masks = {
            'activation': activation_mask,
            'inhibition': inhibition_mask,
            'undefined': undefined_mask
        }
        x_agg = None
        for edge_type, mask in type_masks.items():
            if torch.any(mask):  
                edge_index_type = edge_index[:, mask]
                edge_attr_type = edge_attr[mask, 1:] if edge_attr.size(1) > 1 else None
                x_type = self.conv1_dict[edge_type](x, edge_index_type, edge_attr_type)
                if x_agg is None:
                    x_agg = x_type
                else:
                    x_agg = x_agg + x_type
        if x_agg is None:
            x_agg = torch.zeros(x.size(0), self.hidden_channels * 2, device=x.device)
        x = F.relu(x_agg)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        
        if return_node_features:
            return x
        x = global_mean_pool(x, data.batch if hasattr(data, 'batch') else None)
        return x

class CrossModalAttentionLayer(nn.Module):
    def __init__(self, gene_dim, pathway_dim, attn_dim, dropout=0.2):
        super().__init__()
        self.query_proj = nn.Linear(gene_dim, attn_dim)
        self.key_proj = nn.Linear(pathway_dim, attn_dim)
        self.value_proj = nn.Linear(pathway_dim, attn_dim)
        self.layer_norm1 = nn.LayerNorm(attn_dim)
        self.layer_norm2 = nn.LayerNorm(attn_dim)
        self.dropout = nn.Dropout(dropout)  
        self.softmax = nn.Softmax(dim=-1)
        self.gene_attn_bias = nn.Linear(gene_dim, attn_dim)
        nn.init.xavier_normal_(self.query_proj.weight)
        nn.init.xavier_normal_(self.key_proj.weight)
        self.scale_factor = 1.0 / (attn_dim ** 0.5)
    
    def forward(self, gene_embedding, pathway_node_features):
        batch_size = gene_embedding.size(0)
        num_nodes = pathway_node_features.size(0)
        Q = self.query_proj(gene_embedding)  
        Q = self.layer_norm1(Q).unsqueeze(1)  
        gene_bias = self.gene_attn_bias(gene_embedding) 
        pathway_nodes = pathway_node_features.unsqueeze(0).expand(batch_size, -1, -1)
        K_base = self.key_proj(pathway_nodes) 
        bias_expanded = gene_bias.unsqueeze(1).expand(-1, num_nodes, -1)
        K = self.layer_norm2(K_base + 0.1 * bias_expanded)
        
        V = self.value_proj(pathway_nodes)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale_factor
        if self.training:
            noise = torch.randn_like(attn_scores) * 0.1
            attn_scores = attn_scores + noise
        attn_weights = self.dropout(self.softmax(attn_scores))
        context = torch.bmm(attn_weights, V).squeeze(1)
        
        return context, attn_weights.squeeze(1)

class FusionModel(nn.Module):
    def __init__(self, gene_dim: int = 256, pathway_dim: int = 256, 
                 hidden_dims: list = [256, 128], output_dim: int = 1,  
                 use_gradient_checkpointing: bool = False, seq_len: int = 512,
                 embed_dim: Optional[int] = None, hidden_dim: Optional[int] = None,
                 pathway_data: Optional[object] = None, use_disease_data: bool = True,  
                 disease_emb_dim: int = 32):  
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
                nn.LayerNorm(disease_emb_dim), 
                nn.ReLU(),
                nn.Linear(disease_emb_dim, disease_emb_dim),  
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
        
    def forward(self, gene_seq: torch.Tensor, pathway_data, disease_counts: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gene_embedding = self.gene_enc(gene_seq)
        if gene_embedding.dim() > 2:
            gene_embedding = gene_embedding.mean(dim=1)
        pathway_nodes = self.pathway_enc(pathway_data, return_node_features=True)
        cross_output, attn_weights = self.cross_attn(gene_embedding, pathway_nodes)
        if cross_output.dim() > 2:
            cross_output = cross_output.mean(dim=1)
        
        if self.use_disease_data and disease_counts is not None:
            disease_embedding = self.disease_encoder(disease_counts)
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