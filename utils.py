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


def sample_posterior(posterior_mean, posterior_covariance, n_samples = 10):
    """_summary_

    Args:
        posterior_mean (torch.Size([N, 2])): mean
        posterior_covariance (torch.Size([2 x N, 2 x N])): full covariance matrix
        n_samples (int, optional): Number of samples to draw. Defaults to 10.

    Returns:
        samples (torch.Size([n_samples, N, 2])): _description_
    """
    # print(posterior_mean_long.shape): torch.Size([2 x N])
    posterior_mean_long = posterior_mean.reshape(-1)

    # Add jitter to diagonal for robust cholesky decomposition
    # torch.Size([2 x N, 2 x N])
    posterior_covariance_with_jitter = posterior_covariance + 1e-3 * torch.eye(posterior_covariance.shape[0]).to(posterior_covariance.device)

    # Compute Cholesky decomposition
    # print(L_posterior.shape): torch.Size([2 x N, 2 x N])
    L_posterior = torch.linalg.cholesky(posterior_covariance_with_jitter, upper = False).to(posterior_covariance.device)
    
    # Sample from normal to achieve shape (n_samples, 2 x N) to match covariance matrix
    z = torch.randn(n_samples, posterior_mean_long.shape[0]).to(posterior_covariance.device)

    # Create the samples: posterior_mean + L * z
    # matmul: (n_samples, 2 x N)
    # tiles: # Repeat along first dimension (rows)
    samples_long = posterior_mean_long.tile(n_samples, 1) + torch.matmul(z, L_posterior.mT)

    return samples_long.reshape(n_samples, -1, 2)
