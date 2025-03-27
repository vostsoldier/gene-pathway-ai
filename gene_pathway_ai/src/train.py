import argparse
import csv
import os
from typing import Dict, List, Tuple
import glob
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch_geometric
import numpy as np
from data_loader import load_gene_sequences, load_pathway_graph, load_genes_from_dir
from model import FusionModel
from utils import seq_to_onehot, check_cuda
from visualize import visualize_latent_space

def prepare_data(pos_dir: str, neg_dir: str, pathway_file: str) -> Tuple[DataLoader, DataLoader, torch.Tensor, List[str]]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pathway_data = load_pathway_graph(pathway_file)
    pos_genes = load_genes_from_dir(pos_dir)
    neg_genes = load_genes_from_dir(neg_dir)
    
    print(f"Loaded {len(pos_genes)} positive genes and {len(neg_genes)} negative genes")
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
    train_indices = train_dataset.indices
    train_labels = [labels[idx] for idx in train_indices]
    
    pos_count = sum(1 for l in train_labels if l == 1)
    neg_count = sum(1 for l in train_labels if l == 0)
    if pos_count > 0 and neg_count > 0: 
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

def main(args: Dict) -> None:
    """Main training function"""
    check_cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gene-Pathway Prediction Model")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of training epochs")
    parser.add_argument("--pos_dir", type=str, required=True, help="Directory with positive gene files")
    parser.add_argument("--neg_dir", type=str, required=True, help="Directory with negative gene files")
    parser.add_argument("--pathway", type=str, required=True, help="Path to pathway graph TSV/KGML file")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
    args = parser.parse_args()
    main(args)