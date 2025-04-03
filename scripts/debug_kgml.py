import sys
import xml.etree.ElementTree as ET

def check_kgml_relations(kgml_file):
    """Directly check relations in a KGML file using ElementTree"""
    try:
        tree = ET.parse(kgml_file)
        root = tree.getroot()
        entries = root.findall("./entry")
        relations = root.findall("./relation")
        
        print(f"KGML file: {kgml_file}")
        print(f"Number of entries: {len(entries)}")
        print(f"Number of relations: {len(relations)}")
        if relations:
            print("\nSample relations:")
            for i, rel in enumerate(relations[:5]):
                entry1 = rel.get("entry1")
                entry2 = rel.get("entry2")
                rel_type = rel.get("type", "unknown")
                print(f"  {i+1}. {entry1} -> {entry2} [{rel_type}]")
        else:
            print("\nNo relations found in the KGML file!")
            
        return len(relations) > 0
            
    except Exception as e:
        print(f"Error parsing KGML: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_kgml.py <kgml_file>")
        sys.exit(1)
        
    check_kgml_relations(sys.argv[1])