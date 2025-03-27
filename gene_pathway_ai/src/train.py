import argparse
import csv
import os
from typing import Dict, List, Tuple

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_loader import load_gene_sequences, load_pathway_graph
from model import FusionModel
from utils import seq_to_onehot, check_cuda
from visualize import visualize_latent_space

def prepare_data(gene_files: List[str], pathway_file: str, labels: List[int]) -> Tuple[DataLoader, DataLoader]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pathway_data = load_pathway_graph(pathway_file)
    gene_tensors = []
    for gene_file in gene_files:
        seq = load_gene_sequences(gene_file)
        gene_tensors.append(seq_to_onehot(seq))
    genes_batch = torch.stack(gene_tensors)
    labels_tensor = torch.tensor(labels, dtype=torch.float).view(-1, 1)
    dataset = TensorDataset(genes_batch, labels_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    return train_loader, val_loader, pathway_data

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

def main(args: Dict) -> None:
    check_cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    related_genes = ["TP53", "BRCA2", "RAD51", "PALB2", "CHEK2"]
    unrelated_genes = ["ACTB", "GAPDH", "IL6", "TNF", "APOE"]
    all_gene_files = [args.fasta_path] * (len(related_genes) + len(unrelated_genes))
    labels = [1] * len(related_genes) + [0] * len(unrelated_genes)
    
    print(f"Number of gene files: {len(all_gene_files)}, Number of labels: {len(labels)}")
    train_loader, val_loader, pathway_data = prepare_data(
        all_gene_files, args.kegg_path, labels
    )
    pathway_data = pathway_data.to(device)
    sample_batch = next(iter(train_loader))
    seq_len = sample_batch[0].shape[2]  
    model = FusionModel(seq_len=seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = GradScaler()
    criterion = torch.nn.BCEWithLogitsLoss()
    with open("results/training_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1"])

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    accumulation_steps = 4

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
            all_embeddings, all_labels = gather_latent_space(model, train_loader, device)
            visualize_latent_space(all_embeddings, all_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of training epochs")
    parser.add_argument("--fasta_path", type=str, default="src/data/BRCA1.fasta", help="Path to BRCA1 FASTA")
    parser.add_argument("--kegg_path", type=str, default="src/data/dna_repair_graph.tsv", help="Path to DNA repair data")
    args = parser.parse_args()
    main(args)