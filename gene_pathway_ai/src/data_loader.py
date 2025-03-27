import os
from typing import Tuple
import random

import torch
import pandas as pd
from torch_geometric.data import Data

def load_gene_sequence(fasta_path: str) -> str:
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    with open(fasta_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Skip header, concatenate sequence lines
    seq = "".join([l.strip() for l in lines if not l.startswith(">")])
    return seq

def load_gene_sequences(fasta_path: str, max_length: int = 10000, augment: bool = True) -> str:
    seq = load_gene_sequence(fasta_path)
    seq = seq.upper().replace('U', 'T')  # Convert RNA 'U' to 'T' if present
    
    if augment:
        seq = apply_augmentations(seq, mutation_rate=0.001, deletion_rate=0.0005)
    
    if len(seq) > max_length:
        seq = seq[:max_length]
    else:
        seq += "N" * (max_length - len(seq))
    return seq

def apply_augmentations(seq: str, mutation_rate: float, deletion_rate: float) -> str:
    nucleotides = ['A', 'C', 'G', 'T', 'N']
    seq_list = list(seq)
    new_seq = []
    for base in seq_list:
        # Randomly delete
        if random.random() < deletion_rate:
            continue
        # Randomly mutate
        if random.random() < mutation_rate:
            base = random.choice(nucleotides)
        new_seq.append(base)
    return "".join(new_seq)

def load_pathway_graph(kegg_path: str) -> Data:
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
            go_count = sign_rows['goTerms'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0).mean()
        else:
            go_count = 0.0

        x_list.append([deg, sign_values, go_count])

    x = torch.tensor(x_list, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data