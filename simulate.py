import torch
import numpy as np
import math

# These functions take (non-grid) (N, 2) tensors as input (the x1 and x2 coordinate pairs) and output (N, 2) tensors (u and v i.e. y1 and y2 pairs)

def simulate_convergence(X):
    U = X[:, 1]
    V = X[:, 0]
    # unsqueeze both at last dim and concatenate
    return torch.cat([U.unsqueeze(-1), V.unsqueeze(-1)], dim = -1)

def simulate_merge(X):
    U = (X[:, 1] + 0.5)**2
    V = np.sin(X[:, 0] * math.pi)
    return torch.cat([U.unsqueeze(-1), V.unsqueeze(-1)], dim = -1)

def simulate_branching(X):
    U = X[:, 0] * X[:, 1]
    V = - 0.5 * X[:, 1]**2 + (X[:, 0] - 0.8)
    return torch.cat([U.unsqueeze(-1), V.unsqueeze(-1)], dim = -1)

def simulate_deflection(X):
    U = (X[:, 1] * 6 - 3)**2 + (X[:, 0] * 6 - 3)**2 + 3
    V = -2 * (X[:, 1] * 6 - 3) * (X[:, 0] * 6 - 3)
    return torch.cat([U.unsqueeze(-1), V.unsqueeze(-1)], dim = -1)/10 # divide by 10 to scale down

def simulate_ridge(X):
    U = X[:, 1] + 1
    V = - np.cos(3 * X[:, 0]**3 * math.pi)
    return torch.cat([U.unsqueeze(-1), V.unsqueeze(-1)], dim = -1)