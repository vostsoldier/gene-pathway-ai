import torch
import numpy as np

def seq_to_onehot(seq: str) -> torch.Tensor:
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    arr = [mapping.get(ch, 4) for ch in seq]
    oh = np.eye(5)[arr].T
    return torch.tensor(oh, dtype=torch.float)

def check_cuda() -> None:
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU.")