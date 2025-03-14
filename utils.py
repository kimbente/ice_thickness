import torch
import numpy as np
import random

def set_seed(seed):
    """
    Set the seed for reproducibility.
    """
    # Python random seed
    random.seed(seed)
    
    # NumPy random seed
    np.random.seed(seed)
    
    # PyTorch random seed for CPU and GPU (if available)
    torch.manual_seed(seed)
    
    # For CUDA (if using GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # For deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False