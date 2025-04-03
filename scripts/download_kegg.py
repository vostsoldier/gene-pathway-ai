import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Union, TextIO, Dict
import xml.etree.ElementTree as ET

from Bio.KEGG import REST
from Bio.KEGG.KGML import KGML_parser


def download_kegg_pathway(
    pathway_id: str, 
    output_file: Union[str, Path, None] = None,
    retries: int = 3, 
    delay: int = 1
) -> Path:
    if not pathway_id:
        raise ValueError("Pathway ID cannot be empty")

    if output_file is None:
        output_file = f"{pathway_id}.kgml"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading KEGG pathway: {pathway_id}")
    
    for attempt in range(retries):
        try:
            handle = REST.kegg_get(pathway_id, "kgml")
            with open(output_path, "w") as f:
                f.write(handle.read())
            try:
                validate_kgml(output_path)
                print(f"Successfully downloaded and validated {pathway_id} to {output_path}")
                return output_path
            except Exception as e:
                print(f"Downloaded file is invalid: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                raise
                
        except Exception as e:
            if attempt < retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise ConnectionError(f"Failed to download {pathway_id} after {retries} attempts: {e}")
    
    return output_path  


def validate_kgml(kgml_file: Union[str, Path]) -> bool:
    try:
        pathway = KGML_parser.read(open(kgml_file, "r"))
        if not pathway.entries:
            raise ValueError("KGML file has no entries")
        print(f"KGML validation successful: {len(pathway.entries)} entries, {len(pathway.relations)} relations")
        print(f"Pathway relations count: {len(pathway.relations)}")
        print(f"First few relations: {list(pathway.relations)[:5]}")
        
        return True
    except Exception as e:
        raise ValueError(f"Failed to validate KGML file: {e}")


def count_pathway_elements(kgml_file: Union[str, Path]) -> Dict[str, int]:
    tree = ET.parse(kgml_file)
    root = tree.getroot()
    
    counts = {
        "entries": len(root.findall("./entry")),
        "genes": len([e for e in root.findall("./entry") if e.get("type") == "gene"]),
        "compounds": len([e for e in root.findall("./entry") if e.get("type") == "compound"]),
        "relations": len(root.findall("./relation")),
        "reactions": len(root.findall("./reaction")),
    }
    
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Download KEGG pathway in KGML format and validate it"
    )
    parser.add_argument(
        "-p", "--pathway", 
        default="hsa03440", 
        help="KEGG pathway ID (default: hsa03440 - DNA repair pathway)"
    )
    parser.add_argument(
        "-o", "--output", 
        default=None,
        help="Output file path (default: <pathway_id>.kgml)"
    )
    parser.add_argument(
        "-r", "--retries", 
        type=int, 
        default=3,
        help="Number of retries if download fails (default: 3)"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = download_kegg_pathway(
            args.pathway, 
            args.output or f"{args.pathway}.kgml", 
            args.retries
        )
        counts = count_pathway_elements(output_path)
        print(f"Pathway statistics: {counts}")
        print("\nTo validate the KGML file, run:")
        print(f"python -c \"from Bio.KEGG.KGML import read; print(read(open('{output_path}')))\"\n")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()