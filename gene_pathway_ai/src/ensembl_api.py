import os
import json
import time
import requests

CACHE_FILE = os.path.join(os.path.dirname(__file__), 'ensembl_cache.json')

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def get_gene_disease_associations(gene_id: str, delay: float = 0.5):
    cache = load_cache()
    if gene_id in cache:
        return cache[gene_id]

    url = f"https://rest.ensembl.org/phenotype/gene/homo_sapiens/{gene_id}?content-type=application/json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cache[gene_id] = data
            save_cache(cache)
            time.sleep(delay)
            return data
        else:
            print(f"Error: Received HTTP {response.status_code} for gene {gene_id}.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"API request error for gene {gene_id}: {e}")
        return None