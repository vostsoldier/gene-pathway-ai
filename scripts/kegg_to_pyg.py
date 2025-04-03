import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from Bio.KEGG.KGML import KGML_parser


class KGMLToGraphConverter:
    RELATION_TYPES = {
        "PPrel": 1,  
        "ECrel": 2, 
        "GErel": 3,  
        "PCrel": 4,  
        "maplink": 5, 
        "": 0,  
    }
    
    def __init__(self):
        pass
    
    def is_enzyme(self, entry) -> int:
        if not hasattr(entry, "graphics"):
            return 0
            
        for graphic in entry.graphics:
            if hasattr(graphic, "name") and graphic.name and "EC:" in graphic.name:
                return 1
        if entry.type == "enzyme":
            return 1
            
        return 0
    
    def convert_kgml_to_graph(self, kgml_file: Union[str, Path]) -> Data:
        try:
            pathway = KGML_parser.read(open(kgml_file, "r"))
            G = nx.DiGraph()
            for entry_id, entry in pathway.entries.items():
                G.add_node(entry_id, 
                           name=entry.name, 
                           type=entry.type, 
                           is_enzyme=self.is_enzyme(entry))
            edge_types = []
            for relation in pathway.relations:
                entry1_id = relation._entry1  
                entry2_id = relation._entry2  
                
                if entry1_id in G.nodes and entry2_id in G.nodes:
                    rel_type = relation.type if relation.type else ""
                    G.add_edge(entry1_id, entry2_id, type=rel_type)
                    edge_types.append(rel_type)
            
            degrees = dict(G.degree())
            interaction_counts = {}
            
            for node in G.nodes():
                interaction_counts[node] = len(list(G.successors(node))) + len(list(G.predecessors(node)))
            
            num_nodes = len(G.nodes())
            node_features = torch.zeros((num_nodes, 3), dtype=torch.float)
            
            node_mapping = {node: i for i, node in enumerate(G.nodes())}
            
            for node, idx in node_mapping.items():
                node_data = G.nodes[node]
                node_features[idx, 0] = float(degrees[node])
                node_features[idx, 1] = float(node_data.get('is_enzyme', 0))
                node_features[idx, 2] = float(interaction_counts[node])
            
            edge_index = []
            edge_attr = []
            
            for edge in G.edges(data=True):
                source, target, data = edge
                source_idx = node_mapping[source]
                target_idx = node_mapping[target]
                edge_index.append([source_idx, target_idx])
                
                rel_type = data.get('type', '')
                rel_type_idx = self.RELATION_TYPES.get(rel_type, 0)
                edge_attr.append([rel_type_idx])
            
            if len(edge_index) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 1), dtype=torch.float)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes
            )
            
            data.name = pathway.name
            data.node_mapping = node_mapping
            data.relation_types = edge_types
            
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to convert KGML to graph: {e}")
    
    def save_graph(self, data: Data, output_file: Union[str, Path]):
        torch.save(data, output_file)
        print(f"Graph saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert KEGG KGML files to PyTorch Geometric data objects"
    )
    parser.add_argument("kgml_file", help="Path to the KGML file")
    parser.add_argument(
        "-o", "--output", 
        default=None,
        help="Output file path (default: <input_file>.pt)"
    )
    
    args = parser.parse_args()
    
    try:
        converter = KGMLToGraphConverter()

        data = converter.convert_kgml_to_graph(args.kgml_file)
        output_file = args.output or f"{Path(args.kgml_file).stem}.pt"
        converter.save_graph(data, output_file)
        
        print(f"Conversion successful: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
        print(f"\nNode features [degree, is_enzyme, interaction_count]:")
        print(f"{data.x[:5]}")
        
        if data.edge_attr.shape[0] > 0:
            print(f"\nEdge attributes [relation_type]:")
            print(f"{data.edge_attr[:5]}")
        else:
            print("\nNo edges found in the pathway")
        
        if "hsa03440" in args.kgml_file:
            print("\nTesting assertion for hsa03440:")
            expected_nodes = 52
            try:
                assert data.num_nodes == expected_nodes, f"Expected {expected_nodes} nodes, got {data.num_nodes}"
                print(f"Test passed: {data.num_nodes} nodes found as expected")
            except AssertionError as e:
                print(f"Test failed: {e}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()