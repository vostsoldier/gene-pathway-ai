import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from torch.utils.data import DataLoader 
from typing import List, Optional
try:
    from umap import UMAP  
except ImportError:
    raise ImportError("Please install umap-learn: pip install umap-learn")

def visualize_latent_space(embeddings: np.ndarray, labels: np.ndarray, gene_names=None, filename="results/umap.png"):
    reducer = UMAP(n_neighbors=min(15, len(embeddings)-1), min_dist=0.1, metric='euclidean')
    embeddings_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap='coolwarm',
        alpha=0.8,
        s=100
    )
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, label='Housekeeping Genes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='DNA Repair Genes')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    if gene_names is not None:
        if len(gene_names) > len(embeddings_2d):
            print(f"Warning: {len(gene_names)} gene names provided but only {len(embeddings_2d)} embeddings available")
            gene_names = gene_names[:len(embeddings_2d)]
        
        for i, name in enumerate(gene_names):
            plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                         fontsize=8, alpha=0.7)
    
    plt.title("Gene Embedding Space: DNA Repair vs Housekeeping", fontsize=15)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    os.makedirs("results", exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_attention(attn_weights, gene_names: List[str], pathway_node_names: List[str], 
                        epoch: int, save_dir: str = "results", suffix: str = ""):
    os.makedirs(save_dir, exist_ok=True)
    
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.cpu().numpy()
    if len(attn_weights.shape) > 2:
        attn_weights = attn_weights.mean(axis=1)
    
    if len(gene_names) > attn_weights.shape[0]:
        print(f"Warning: {len(gene_names)} gene names provided but only {attn_weights.shape[0]} attention weight vectors available")
        gene_names = gene_names[:attn_weights.shape[0]]  
    max_genes = min(len(gene_names), 20, attn_weights.shape[0])
    cosine_similarities = np.zeros((max_genes, max_genes))
    for i in range(max_genes):
        for j in range(max_genes):
            if i == j:
                cosine_similarities[i,j] = 1.0
                continue
            sim = np.dot(attn_weights[i], attn_weights[j]) / (
                np.linalg.norm(attn_weights[i]) * np.linalg.norm(attn_weights[j]) + 1e-8)
            cosine_similarities[i,j] = sim
    
    avg_similarity = (cosine_similarities.sum() - max_genes) / (max_genes * (max_genes - 1))
    print(f"Average similarity between gene attention patterns: {avg_similarity:.4f}")
    if avg_similarity > 0.9:
        print("WARNING: Gene attention patterns are nearly identical! Model may need more training.")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_similarities, xticklabels=gene_names[:max_genes], 
                yticklabels=gene_names[:max_genes], cmap='coolwarm')
    plt.title(f"Gene Attention Pattern Similarity (Epoch {epoch})\nAvg: {avg_similarity:.4f}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/attention_similarity{suffix}.png", dpi=300)
    plt.close()
    normalized_weights = attn_weights.copy()
    for i in range(attn_weights.shape[0]):
        row_min = normalized_weights[i].min()
        row_max = normalized_weights[i].max()
        if row_max > row_min:  
            normalized_weights[i] = (normalized_weights[i] - row_min) / (row_max - row_min)

    max_nodes = min(len(pathway_node_names), 30)
    display_genes = gene_names[:max_genes]
    display_nodes = pathway_node_names[:max_nodes]
    display_weights = normalized_weights[:max_genes, :max_nodes]
    plt.figure(figsize=(14, 10)) 
    ax = sns.heatmap(
        display_weights,
        xticklabels=display_nodes,
        yticklabels=display_genes,
        cmap='viridis',
        annot=False,
        linewidths=0.5
    )
    plt.title(f"Gene-Pathway Attention Weights (Epoch {epoch})")
    plt.xlabel("Pathway Nodes")
    plt.ylabel("Genes")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/attention_heatmap{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()
    grid_cols = min(5, max_genes)
    grid_rows = (max_genes + grid_cols - 1) // grid_cols  
    
    plt.figure(figsize=(16, 3*grid_rows))
    for i, gene in enumerate(display_genes):
        gene_weights = attn_weights[i]
        top_indices = np.argsort(gene_weights)[-5:][::-1]
        top_nodes = [pathway_node_names[idx] for idx in top_indices]
        top_weights = [gene_weights[idx] for idx in top_indices]
        plt.subplot(grid_rows, grid_cols, i+1)
        plt.barh(top_nodes, top_weights, color=plt.cm.viridis(np.linspace(0.2, 0.8, 5)))
        plt.title(f"{gene}", fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
    
    plt.suptitle(f"Top 5 Pathway Nodes by Gene (Epoch {epoch})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.savefig(f"{save_dir}/attention_top5{suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()
    if max_genes > 10:
        for split_idx, split_range in enumerate([range(0, 10), range(10, max_genes)]):
            plt.figure(figsize=(16, 15))
            for i, idx in enumerate(split_range):
                if idx >= max_genes:
                    break
                gene = display_genes[idx]
                gene_weights = attn_weights[idx]
                top_indices = np.argsort(gene_weights)[-5:][::-1]
                top_nodes = [pathway_node_names[j] for j in top_indices]
                top_weights = [gene_weights[j] for j in top_indices]
                
                plt.subplot(5, 2, i+1)
                plt.barh(top_nodes, top_weights, color=plt.cm.viridis(np.linspace(0.2, 0.8, 5)))
                plt.title(f"{gene}", fontsize=11)
                plt.xticks(fontsize=9)
                plt.yticks(fontsize=9)
            
            plt.suptitle(f"Top 5 Pathway Nodes by Gene - Set {split_idx+1} (Epoch {epoch})", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f"{save_dir}/attention_top5_set{split_idx+1}{suffix}.png", dpi=300, bbox_inches='tight')
            plt.close()
def attention_heatmap(attn_weights, gene_names, node_names, filename="heatmap.png"):
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.cpu().numpy()
        
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        attn_weights,
        xticklabels=node_names[:attn_weights.shape[1]],
        yticklabels=gene_names[:attn_weights.shape[0]],
        cmap='viridis'
    )
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def visualize_attention_for_all_genes(model, dataset, device, pathway_data, pathway_node_names, gene_names, epoch, save_dir="results"):
    model.eval()
    full_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    with torch.no_grad():
        for genes, disease_counts, labels in full_loader:
            genes = genes.to(device)
            disease_counts = disease_counts.to(device)
            _, attn_weights, _ = model(genes, pathway_data, disease_counts=disease_counts)
            
            print(f"Creating attention visualization for all {len(genes)} genes")
            print(f"Gene names: {', '.join(gene_names)}")
            print(f"Attention weights shape: {attn_weights.shape}")
            print(f"Number of genes in batch: {len(genes)} / Number of gene names: {len(gene_names)}")
            visualize_attention(
                attn_weights,
                gene_names,  
                pathway_node_names,
                epoch,
                save_dir=save_dir
            )
            break

def visualize_attention_for_original_genes(model, dataset, device, pathway_data, pathway_node_names, gene_names, epoch, save_dir="results"):
    model.eval()
    original_genes = []
    original_indices = []
    
    for i, name in enumerate(gene_names):
        if "_aug" not in name:
            original_genes.append(name)
            original_indices.append(i)
    
    print(f"Found {len(original_genes)} original genes (without augmentations)")
    selected_data = []
    for idx in original_indices:
        if idx < len(dataset):
            selected_data.append(dataset[idx])
    if not selected_data:
        print("Processing full dataset and filtering results...")
        full_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        
        with torch.no_grad():
            for genes, disease_counts, labels in full_loader:
                genes = genes.to(device)
                disease_counts = disease_counts.to(device)
                _, attn_weights, _ = model(genes, pathway_data, disease_counts=disease_counts)
                original_attn = []
                for i in original_indices:
                    if i < len(genes):
                        original_attn.append(attn_weights[i].unsqueeze(0))
                
                if original_attn:
                    original_attn = torch.cat(original_attn, dim=0)
                    print(f"Creating attention visualization for {len(original_genes)} original genes")
                    visualize_attention(
                        original_attn,
                        original_genes,  
                        pathway_node_names,
                        epoch,
                        save_dir=save_dir,
                        suffix="_original_only"
                    )
                break
    else:
        genes_batch, disease_counts_batch, labels_batch = zip(*selected_data)
        genes_tensor = torch.stack(genes_batch).to(device)
        disease_counts_tensor = torch.stack(disease_counts_batch).to(device)
        
        with torch.no_grad():
            _, attn_weights, _ = model(genes_tensor, pathway_data, disease_counts=disease_counts_tensor)
            
            print(f"Creating attention visualization for {len(original_genes)} original genes")
            visualize_attention(
                attn_weights,
                original_genes,  
                pathway_node_names,
                epoch,
                save_dir=save_dir,
                suffix="_original_only"
            )