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

def make_grid(n_side, start = 0.0, end = 1.0):
    """ Make a grid of points in 2D space using torch

    Args:
        n_side (torch.Size([ ]) i.e. scalar): This is the same as H == W == grid_size
        start (torch.Size([ ]) i.e. scalar, optional): Staring point of both x and y. Defaults to 0.0.
        end (torch.Size([ ]) i.e. scalar, optional): End point of both x and y. Defaults to 1.0.
    Returns:
        x_test_grid (torch.Size([n_side, n_side, 2])): 2D grid of points 
        x_test_long (torch.Size([n_side * n_side, 2])): flat version of the grid
    """
    side_array = torch.linspace(start = start, end = end, steps = n_side)
    XX, YY = torch.meshgrid(side_array, side_array, indexing = "xy")
    x_test_grid = torch.cat([XX.unsqueeze(-1), YY.unsqueeze(-1)], dim = -1)
    x_test_long = x_test_grid.reshape(-1, 2)
    
    return x_test_grid, x_test_long