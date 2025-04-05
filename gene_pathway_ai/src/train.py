import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from typing import Dict, List, Tuple
import glob
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler, ConcatDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch_geometric
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import seaborn as sns  
from data_loader import load_gene_sequences, load_pathway_graph, load_genes_from_dir
from model import FusionModel
from utils import seq_to_onehot, check_cuda
from visualize import visualize_latent_space, visualize_attention, visualize_attention_for_all_genes
from losses import HybridLoss

def prepare_data(pos_dir: str, neg_dir: str, pathway_file: str = None, preloaded_pathway_data = None) -> Tuple[DataLoader, DataLoader, torch.Tensor, List[str], torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from ensembl_api import get_gene_disease_associations
    
    if preloaded_pathway_data is not None:
        pathway_data = preloaded_pathway_data
    elif pathway_file is not None:
        pathway_data = load_pathway_graph(pathway_file)
    else:
        raise ValueError("Either pathway_file or preloaded_pathway_data must be provided")
    
    print(f"Loading positive genes with augmentation...")
    pos_genes = load_genes_from_dir(pos_dir, max_length=10000, augment=True, num_augmentations=10)
    
    print(f"Loading negative genes with augmentation...")
    neg_genes = load_genes_from_dir(neg_dir, max_length=10000, augment=True, num_augmentations=10)
    
    print(f"Loaded {len(pos_genes)} positive genes and {len(neg_genes)} negative genes")
    print(f"Original count: ~{len(pos_genes)//10} positive, ~{len(neg_genes)//10} negative")
    pos_ratio = len(pos_genes) / (len(pos_genes) + len(neg_genes))
    print(f"Class distribution: {pos_ratio:.2f} positive, {1-pos_ratio:.2f} negative")

    if abs(pos_ratio - 0.5) > 0.1:  
        print("Warning: Class imbalance detected (>10% difference)")
        print("Using weighted sampling to balance classes")
        use_class_weighting = True
    else:
        print("Classes are reasonably balanced")
        use_class_weighting = False
    
    all_original_genes = set()
    for gene_name, _ in pos_genes + neg_genes:
        base_name = gene_name.split('_aug')[0]
        all_original_genes.add(base_name)
    
    print(f"Fetching disease associations from ENSEMBL for {len(all_original_genes)} original genes...")
    gene_disease_map = {}
    for gene_name in tqdm(all_original_genes, desc="Querying ENSEMBL API"):
        associations = get_gene_disease_associations(gene_name)
        if associations is not None and isinstance(associations, list):
            count = len(associations)
            diseases = []
            for assoc in associations:
                if isinstance(assoc, dict) and 'description' in assoc:
                    disease_name = assoc['description']
                    diseases.append(disease_name)
        else:
            count = 0
            diseases = ["Unknown"]
            
        gene_disease_map[gene_name] = {
            'count': count,
            'diseases': diseases
        }
    
    gene_tensors = []
    gene_names = []
    labels = []
    disease_counts = []
    gene_disease_data = []
    
    for gene_name, seq in pos_genes + neg_genes:
        base_name = gene_name.split('_aug')[0]  
        assert ">" not in seq, f"Header character '>' found in sequence from {gene_name}"
        assert len(seq) == 10000, f"Sequence length mismatch for {gene_name}: {len(seq)} != 10000"
        
        gene_tensors.append(seq_to_onehot(seq))
        gene_names.append(gene_name)
        
        disease_data = gene_disease_map[base_name]
        disease_counts.append([disease_data['count']])
        
        gene_disease_data.append({
            'gene': gene_name,
            'original_gene': base_name,
            'diseases': ';'.join(disease_data['diseases']) if disease_data['diseases'] else "None"
        })
    
    for _ in pos_genes:
        labels.append(1)
    for _ in neg_genes:
        labels.append(0)
        
    import pandas as pd
    os.makedirs('data', exist_ok=True)
    gene_disease_df = pd.DataFrame(gene_disease_data)
    gene_disease_df.to_csv('data/gene_disease_associations.csv', index=False)
    print(f"Exported gene-disease associations to data/gene_disease_associations.csv")
    
    genes_batch = torch.stack(gene_tensors)
    labels_tensor = torch.tensor(labels, dtype=torch.float).view(-1, 1)
    disease_counts_tensor = torch.tensor(disease_counts, dtype=torch.float)
    
    dataset = TensorDataset(genes_batch, disease_counts_tensor, labels_tensor)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) 
    )
    if use_class_weighting:
        train_indices = train_dataset.indices
        train_labels = [labels[idx] for idx in train_indices]
        
        pos_count = sum(1 for l in train_labels if l == 1)
        neg_count = sum(1 for l in train_labels if l == 0)
        class_weights = [1.0/neg_count if l == 0 else 1.0/pos_count for l in train_labels]
        sample_weights = torch.FloatTensor(class_weights)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    return train_loader, val_loader, pathway_data, gene_names

def train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, pathway_data, accumulation_steps):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    
    for i, (genes, disease_counts, labels) in enumerate(train_loader):
        genes, disease_counts, labels = genes.to(device), disease_counts.to(device), labels.to(device)
        with autocast():
            predictions, attn_weights, latent = model(genes, pathway_data, disease_counts=disease_counts)
            loss = criterion(predictions, labels, attn_weights)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item() * genes.size(0)
        
    return total_loss / len(train_loader.dataset)

def evaluate(model, val_loader, device, pathway_data, criterion, epoch=None, pathway_node_names=None, gene_names=None, 
             is_best=False, final_epoch=False):
    model.eval()
    preds, true_labels = [], []
    total_loss = 0.0
    all_gene_indices = []
    all_attn_weights = []

    with torch.no_grad():
        batch_start_idx = 0
        for batch_idx, (genes, disease_counts, labels) in enumerate(val_loader):
            genes, disease_counts, labels = genes.to(device), disease_counts.to(device), labels.to(device)
            predictions, attn_weights, _ = model(genes, pathway_data, disease_counts=disease_counts)
            loss = criterion(predictions, labels, attn_weights)
            total_loss += loss.item() * genes.size(0)
            all_attn_weights.append(attn_weights.cpu())
            batch_size = len(genes)
            batch_indices = list(range(batch_start_idx, batch_start_idx + batch_size))
            all_gene_indices.extend(batch_indices)
            batch_start_idx += batch_size
            
            predicted = (torch.sigmoid(predictions) > 0.5).cpu().numpy()
            preds.extend(predicted)
            true_labels.extend(labels.cpu().numpy())
    
    val_loss = total_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, zero_division=0)
    recall = recall_score(true_labels, preds, zero_division=0)
    f1 = f1_score(true_labels, preds, zero_division=0)
    
    return val_loss, accuracy, precision, recall, f1

def gather_latent_space(model, data_loader, device, pathway_data=None):
    if pathway_data is None:
        for batch in data_loader:
            if len(batch) > 2 and isinstance(batch[2], torch_geometric.data.Data):
                pathway_data = batch[2]
                break
    
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for genes, disease_counts, labels in data_loader: 
            genes = genes.to(device)
            disease_counts = disease_counts.to(device)
            predictions, attn_weights, latent = model(genes, pathway_data, disease_counts=disease_counts)
            
            all_embeddings.append(latent.cpu().numpy())
            all_labels.append(labels.numpy())
            
    all_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    
    return all_embeddings, all_labels

def visualize_all_genes(model, train_loader, val_loader, device, pathway_data, gene_names):
    print("Creating full gene comparison visualization...")
    full_dataset = ConcatDataset([
        train_loader.dataset, 
        val_loader.dataset
    ])
    full_loader = DataLoader(full_dataset, batch_size=len(full_dataset))
    
    model.eval()
    with torch.no_grad():
        for genes, disease_counts, labels in full_loader:  
            genes = genes.to(device)
            disease_counts = disease_counts.to(device)
            _, _, combined = model(genes, pathway_data, disease_counts=disease_counts)
            
            embeddings = combined.cpu().numpy()
            label_values = labels.cpu().numpy()
            
            visualize_latent_space(
                embeddings, 
                label_values, 
                gene_names=gene_names,
                filename="results/all_genes_umap.png"
            )
            break

def visualize_all_gene_attention(model, train_loader, val_loader, device, pathway_data, 
                                pathway_node_names, gene_names, epoch, save_dir="results", suffix=""):
    print("Creating comprehensive attention visualization for all genes...")
    
    full_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    full_loader = DataLoader(full_dataset, batch_size=len(full_dataset), shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for genes, disease_counts, labels in full_loader:
            genes = genes.to(device)
            disease_counts = disease_counts.to(device)
            _, attn_weights, _ = model(genes, pathway_data, disease_counts=disease_counts)
            
            print(f"Visualizing attention for all {len(genes)} genes")
            visualize_attention(
                attn_weights,
                gene_names,
                pathway_node_names,
                epoch,
                save_dir=save_dir,
                suffix=suffix
            )
            break

def main(args: Dict, preloaded_pathway_data=None) -> None:
    check_cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    if preloaded_pathway_data is not None:
        pathway_data = preloaded_pathway_data
        train_loader, val_loader, _, gene_names = prepare_data(
            args.pos_dir, args.neg_dir, None, preloaded_pathway_data
        )
    else:
        train_loader, val_loader, pathway_data, gene_names = prepare_data(
            args.pos_dir, args.neg_dir, args.pathway
        )
    pathway_data = pathway_data.to(device)
    sample_batch = next(iter(train_loader))
    seq_len = sample_batch[0].shape[2]
    pathway_feat_dim = pathway_data.x.shape[1]  
    model = FusionModel(
        seq_len=seq_len, 
        pathway_dim=pathway_feat_dim,
        embed_dim=64,
        hidden_dim=64
    ).to(device)
    
    print(f"Model initialized with sequence length {seq_len} and {pathway_feat_dim} pathway features")
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()
    criterion = HybridLoss()
    pathway_node_names = []
    if hasattr(pathway_data, 'node_names'):
        pathway_node_names = pathway_data.node_names
    elif hasattr(pathway_data, 'names'):
        pathway_node_names = pathway_data.names
    else:
        pathway_node_names = [f"Node_{i+1}" for i in range(pathway_data.x.shape[0])]
    
    with open("results/training_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1"])

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5 
    is_best = False
    
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, pathway_data, args.accumulation_steps)
        is_final = (epoch == args.epochs - 1)
        
        val_loss, accuracy, precision, recall, f1 = evaluate(
            model, val_loader, device, pathway_data, criterion, 
            epoch=epoch, pathway_node_names=pathway_node_names, gene_names=gene_names,
            is_best=is_best, final_epoch=is_final
        )
        with open(os.path.join(args.output_dir, 'training_log.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, accuracy, precision, recall, f1])
        
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            is_best = True
            
            if pathway_node_names is not None and gene_names is not None:
                print("Generating visualizations for best model...")
                visualize_all_gene_attention(
                    model,
                    train_loader,
                    val_loader,
                    device, 
                    pathway_data,
                    pathway_node_names,
                    gene_names,
                    epoch,
                    save_dir="results",
                    suffix="_best"
                )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        if epoch % 5 == 0:  
            all_embeddings, all_labels = gather_latent_space(model, val_loader, device, pathway_data)
            visualize_latent_space(all_embeddings, all_labels, gene_names=gene_names)
            if not pathway_node_names:  
                pathway_node_names = [f"Node_{i}" for i in range(pathway_data.x.shape[0])]
                
            visualize_all_gene_attention(
                model, train_loader, val_loader, device, 
                pathway_data, pathway_node_names, gene_names, epoch
            )
    evaluate(
        model, val_loader, device, pathway_data, criterion, 
        epoch=args.epochs, pathway_node_names=pathway_node_names, gene_names=gene_names,
        is_best=False, final_epoch=True
    )
    visualize_all_genes(model, train_loader, val_loader, device, pathway_data, gene_names)

def create_final_visualization(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.pathway and os.path.exists(args.pathway):
        if args.pathway.endswith('.kgml'):
            print(f"Loading KEGG pathway from file: {args.pathway}")
            from data_loader import load_kegg_pathway
            pathway_data = load_kegg_pathway(local_file=args.pathway)
        else:
            print(f"Loading pathway from TSV: {args.pathway}")
            pathway_data = load_pathway_graph(args.pathway)
    else:
        print(f"Downloading KEGG pathway for visualization: {args.kegg_id}")
        from data_loader import load_kegg_pathway
        pathway_data = load_kegg_pathway(pathway_id=args.kegg_id)
    train_loader, val_loader, _, gene_names = prepare_data(
        args.pos_dir,
        args.neg_dir,
        None, 
        pathway_data
    )
    pathway_data = pathway_data.to(device)
    best_model = FusionModel(
        seq_len=10000,
        embed_dim=64,    
        hidden_dim=64, 
        pathway_dim=4  
    ).to(device)
    best_model.load_state_dict(torch.load("results/best_model.pt"))
    print("Creating final visualization with best model...")
    visualize_all_genes(
        best_model, 
        train_loader, 
        val_loader, 
        device, 
        pathway_data, 
        gene_names
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gene-Pathway Prediction Model")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of training epochs")
    parser.add_argument("--pos_dir", type=str, required=True, help="Directory with positive gene files")
    parser.add_argument("--neg_dir", type=str, required=True, help="Directory with negative gene files")
    parser.add_argument("--pathway", type=str, help="Path to pathway graph file (KGML or TSV)")
    parser.add_argument("--kegg_id", type=str, default="hsa03440", help="KEGG pathway ID to download")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save outputs")
    
    args = parser.parse_args()
    pathway_data = None
    if args.pathway and os.path.exists(args.pathway):
        if args.pathway.endswith('.kgml'):
            print(f"Loading KEGG pathway from file: {args.pathway}")
            from data_loader import load_kegg_pathway
            pathway_data = load_kegg_pathway(local_file=args.pathway)
        else:
            print(f"Loading pathway from TSV: {args.pathway}")
            pathway_data = load_pathway_graph(args.pathway)
    else:
        print(f"Downloading KEGG pathway: {args.kegg_id}")
        from data_loader import load_kegg_pathway
        pathway_data = load_kegg_pathway(pathway_id=args.kegg_id)
    main(args, preloaded_pathway_data=pathway_data)
    create_final_visualization(args)