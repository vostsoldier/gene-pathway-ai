import argparse
import csv
import os
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
from visualize import visualize_latent_space

def prepare_data(pos_dir: str, neg_dir: str, pathway_file: str = None, preloaded_pathway_data = None) -> Tuple[DataLoader, DataLoader, torch.Tensor, List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preloaded_pathway_data is not None:
        pathway_data = preloaded_pathway_data
    elif pathway_file is not None:
        pathway_data = load_pathway_graph(pathway_file)
    else:
        raise ValueError("Either pathway_file or preloaded_pathway_data must be provided")
    pos_files = glob.glob(os.path.join(pos_dir, "*.txt")) + glob.glob(os.path.join(pos_dir, "*.fasta"))
    neg_files = glob.glob(os.path.join(neg_dir, "*.txt")) + glob.glob(os.path.join(neg_dir, "*.fasta"))
    
    print(f"Found {len(pos_files)} positive gene files and {len(neg_files)} negative gene files")
    pos_genes = []
    print("Loading positive genes...")
    for file_path in tqdm(pos_files, desc="Loading DNA repair genes"):
        gene_name = os.path.splitext(os.path.basename(file_path))[0].upper()
        seq = load_gene_sequences(file_path, max_length=10000, augment=True)
        pos_genes.append((gene_name, seq))
    neg_genes = []
    print("Loading negative genes...")
    for file_path in tqdm(neg_files, desc="Loading housekeeping genes"):
        gene_name = os.path.splitext(os.path.basename(file_path))[0].upper()
        seq = load_gene_sequences(file_path, max_length=10000, augment=True)
        neg_genes.append((gene_name, seq))
    
    print(f"Loaded {len(pos_genes)} positive genes and {len(neg_genes)} negative genes")
    pos_ratio = len(pos_genes) / (len(pos_genes) + len(neg_genes))
    print(f"Class distribution: {pos_ratio:.2f} positive, {1-pos_ratio:.2f} negative")
    if abs(pos_ratio - 0.5) > 0.1:
        print("Warning: Class imbalance detected (>10% difference)")
        print("Using weighted sampling to balance classes")
        use_class_weighting = True
    else:
        print("Classes are reasonably balanced")
        use_class_weighting = False
    gene_tensors = []
    gene_names = []
    labels = []
    
    for gene_name, seq in pos_genes:
        assert ">" not in seq, f"Header character '>' found in sequence from {gene_name}"
        assert len(seq) == 10000, f"Sequence length mismatch for {gene_name}: {len(seq)} != 10000"
        
        gene_tensors.append(seq_to_onehot(seq))
        gene_names.append(gene_name)
        labels.append(1)  
    for gene_name, seq in neg_genes:
        assert ">" not in seq, f"Header character '>' found in sequence from {gene_name}"
        assert len(seq) == 10000, f"Sequence length mismatch for {gene_name}: {len(seq)} != 10000"
        
        gene_tensors.append(seq_to_onehot(seq))
        gene_names.append(gene_name)
        labels.append(0)  
    genes_batch = torch.stack(gene_tensors)
    labels_tensor = torch.tensor(labels, dtype=torch.float).view(-1, 1)
    dataset = TensorDataset(genes_batch, labels_tensor)
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
    step = 0
    total_loss = 0.0
    
    for i, (genes, labels) in enumerate(train_loader):
        genes, labels = genes.to(device), labels.to(device)
        with autocast():
            outputs = model(genes, pathway_data)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * genes.size(0)
        
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        step += 1

    avg_loss = total_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
    return avg_loss

def evaluate(model, val_loader, device, pathway_data, criterion):
    model.eval()
    preds, true_labels = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for genes, labels in val_loader:
            genes = genes.to(device)
            labels = labels.to(device)
            
            out = model(genes, pathway_data)
            loss = criterion(out, labels)
            total_loss += loss.item() * genes.size(0)
            
            predicted = (torch.sigmoid(out) > 0.5).cpu().numpy()
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
        for genes, labels in data_loader:
            genes = genes.to(device)
            gene_embed = model.gene_enc(genes)
            path_embed = model.pathway_enc(pathway_data)
            path_embed = path_embed.repeat(gene_embed.size(0), 1)
            combined = torch.cat([gene_embed, path_embed], dim=1)
            all_embeddings.append(combined.cpu().numpy())
            all_labels.append(labels.numpy())
    all_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    
    return all_embeddings, all_labels

def visualize_all_genes(model, train_loader, val_loader, device, pathway_data, gene_names):
    print("Creating full gene comparison visualization...")
    
    # Create a dataset with all genes
    full_dataset = ConcatDataset([
        train_loader.dataset, 
        val_loader.dataset
    ])
    full_loader = DataLoader(full_dataset, batch_size=len(full_dataset))
    
    model.eval()
    with torch.no_grad():
        for genes, labels in full_loader:
            genes = genes.to(device)
            gene_embed = model.gene_enc(genes)
            path_embed = model.pathway_enc(pathway_data)
            path_embed = path_embed.repeat(gene_embed.size(0), 1)
            combined = torch.cat([gene_embed, path_embed], dim=1)
            embeddings = combined.cpu().numpy()
            label_values = labels.cpu().numpy()
            visualize_latent_space(
                embeddings, 
                label_values, 
                gene_names=gene_names,
                filename="results/all_genes_umap.png"
            )
            break

def main(args: Dict, preloaded_pathway_data=None) -> None:
    """Main training function"""
    check_cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    if preloaded_pathway_data is not None:
        pathway_data = preloaded_pathway_data
        train_loader, val_loader, _, gene_names = prepare_data(
            args.pos_dir, 
            args.neg_dir, 
            None, 
            preloaded_pathway_data
        )
    else:
        train_loader, val_loader, pathway_data, gene_names = prepare_data(
            args.pos_dir,
            args.neg_dir,
            args.pathway
        )
    pathway_data = pathway_data.to(device)
    
    sample_batch = next(iter(train_loader))
    seq_len = sample_batch[0].shape[2] 
    model = FusionModel(seq_len=seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = GradScaler()
    criterion = torch.nn.BCEWithLogitsLoss()  
    with open("results/training_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1"])

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    accumulation_steps = args.accumulation_steps

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, pathway_data, accumulation_steps)
        val_loss, accuracy, precision, recall, f1 = evaluate(model, val_loader, device, pathway_data, criterion)
        with open("results/training_log.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, accuracy, precision, recall, f1])
        
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "results/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        if epoch % 5 == 0:  
            all_embeddings, all_labels = gather_latent_space(model, val_loader, device, pathway_data)
            visualize_latent_space(all_embeddings, all_labels, gene_names=gene_names)
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
    best_model = FusionModel(seq_len=10000).to(device)
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