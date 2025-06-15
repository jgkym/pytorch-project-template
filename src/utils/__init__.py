import numpy as np
import torch


def set_seed(config) -> None:
    """Sets the seed for reproducibility."""
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if "cuda" in str(config.device):
        torch.cuda.manual_seed(config.random_seed)
    elif "mps" in str(config.device):
        torch.mps.manual_seed(config.random_seed)
    print(f"Seed set to {config.random_seed}")
