import numpy as np
import matplotlib.pyplot as plt
try:
    from umap import UMAP  
except ImportError:
    raise ImportError("Please install umap-learn: pip install umap-learn")

def visualize_latent_space(embeddings: np.ndarray, labels: np.ndarray):
    """
    Applies UMAP to latent embeddings and saves a 2D scatter plot to results/umap.png.
    embeddings shape: [num_samples, embed_dim]
    labels shape: [num_samples], e.g. class labels or metadata
    """
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    embeddings_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap='viridis',
        alpha=0.8
    )
    plt.colorbar(scatter, label='Class Label')
    plt.title("UMAP Latent Space")
    plt.savefig("results/umap.png", dpi=150)
    plt.close()