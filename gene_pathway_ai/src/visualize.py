import numpy as np
import matplotlib.pyplot as plt
import os
try:
    from umap import UMAP  
except ImportError:
    raise ImportError("Please install umap-learn: pip install umap-learn")

def visualize_latent_space(embeddings: np.ndarray, labels: np.ndarray, gene_names=None):
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
    plt.savefig("results/umap.png", dpi=300, bbox_inches='tight')
    plt.close()