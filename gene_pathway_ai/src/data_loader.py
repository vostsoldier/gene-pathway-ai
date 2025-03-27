import os
import random
from typing import List, Tuple
from tqdm import tqdm

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

def load_gene_sequence(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Gene file not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    seq = "".join([l.strip() for l in lines if not l.startswith(">")])
    return seq

def load_genes_from_dir(dir_path: str, max_length: int = 10000, augment: bool = True) -> List[Tuple[str, str]]:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    gene_files = [f for f in os.listdir(dir_path) if f.endswith(('.txt', '.fasta'))]
    result = []
    
    print(f"Loading {len(gene_files)} genes from {dir_path}")
    for file_name in tqdm(gene_files, desc="Loading genes"):
        file_path = os.path.join(dir_path, file_name)
        gene_name = os.path.splitext(file_name)[0].upper()
        seq = load_gene_sequences(file_path, max_length=max_length, augment=augment)
        
        result.append((gene_name, seq))
    
    return result

def load_gene_sequences(file_path: str, max_length: int = 10000, augment: bool = True) -> str:
    seq = load_gene_sequence(file_path)
    
    seq = seq.upper().replace('U', 'T') 
    assert ">" not in seq, f"Header character '>' found in sequence from {file_path}"
    if augment:
        seq = apply_augmentations(seq, mutation_rate=0.001, deletion_rate=0.0005)
    if len(seq) > max_length:
        seq = seq[:max_length]
    else:
        seq += "N" * (max_length - len(seq))
    assert len(seq) == max_length, f"Sequence length mismatch: {len(seq)} != {max_length}"
    
    return seq

def apply_augmentations(seq: str, mutation_rate: float, deletion_rate: float) -> str:
    nucleotides = ['A', 'C', 'G', 'T']
    seq_list = list(seq)
    new_seq = []
    
    for base in seq_list:
        if random.random() < deletion_rate:
            continue
        if random.random() < mutation_rate:
            base = random.choice(nucleotides)
            
        new_seq.append(base)
    
    return "".join(new_seq)

def load_pathway_graph(kegg_path: str) -> Data:
    if not os.path.exists(kegg_path):
        raise FileNotFoundError(f"Pathway file not found: {kegg_path}")
        
    df = pd.read_csv(kegg_path, sep='\t')
    nodes = list(set(df['source']).union(df['target']))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    degrees = {n: 0 for n in nodes}
    edges = []
    for _, row in df.iterrows():
        src = node_to_idx[row['source']]
        tgt = node_to_idx[row['target']]
        edges.append([src, tgt])
        degrees[row['source']] += 1
        degrees[row['target']] += 1
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x_list = []
    for node in nodes:
        deg = degrees[node]
        sign_rows = df[(df['source'] == node) | (df['target'] == node)]
        if 'interactionType' in sign_rows.columns:
            sign_values = sign_rows['interactionType'].apply(
                lambda x: 1 if x == 'activation' else (-1 if x == 'inhibition' else 0)
            ).mean()
        else:
            sign_values = 0.0
        if 'goTerms' in sign_rows.columns and not sign_rows['goTerms'].isnull().all():
            go_count = sign_rows['goTerms'].apply(
                lambda x: len(str(x).split(',')) if pd.notnull(x) else 0
            ).mean()
        else:
            go_count = 0.0
            
        x_list.append([deg, sign_values, go_count])
    x = torch.tensor(x_list, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data