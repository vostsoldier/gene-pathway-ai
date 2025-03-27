# Gene-Pathway Mapping AI

A deep learning framework to predict associations between genes and biological pathways, specifically discriminating DNA repair genes from housekeeping genes.

## Overview

This project uses a dual-encoder neural network architecture to:
1. Process gene sequences using convolutional networks
2. Process pathway graphs using graph neural networks 
3. Combine these representations to predict pathway membership

## Installation

```bash
# Clone repository
git clone https://github.com/vostsoldier/secret.git
cd secret/gene_pathway_ai

# Install dependencies
pip install -r [requirements.txt](http://_vscodecontentref_/6)

# Basic training with KEGG pathway (downloads hsa03440 DNA repair pathway)
python src/train.py \
  --pos_dir src/data/pos_genes \
  --neg_dir src/data/neg_genes \
  --kegg_id hsa03440

# Training with local KGML file  
python src/train.py \
  --pos_dir src/data/pos_genes \
  --neg_dir src/data/neg_genes \
  --pathway src/data/dna_repair.kgml

# Training with local TSV graph file
python src/train.py \
  --pos_dir src/data/pos_genes \
  --neg_dir src/data/neg_genes \
  --pathway src/data/dna_repair_graph.tsv

  # With additional parameters
python src/train.py \
  --pos_dir src/data/pos_genes \
  --neg_dir src/data/neg_genes \
  --kegg_id hsa03440 \
  --epochs 100 \
  --lr 0.0005 \
  --accumulation_steps 2