import torch
import numpy as np
import random
import warnings

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

def compute_divergence_field(mean_pred, x_grad):
    """_summary_

    Args:
        mean_pred (torch.Size(N, 2)): _description_
        x_grad (torch.Size(N, 2)): _description_

    Returns:
        torch.Size(N, 1): The div field is scalar because we add the two components
    """
    # Because autograd computes gradients of the output w.r.t. the inputs...
    # ... we specify which component of the output you want the gradient of via masking
    # mean_pred is a vector values output
    u_indicator, v_indicator = torch.zeros_like(mean_pred), torch.zeros_like(mean_pred)

    # output mask
    u_indicator[:, 0] = 1.0 # output column u selected
    v_indicator[:, 1] = 1.0 # output column v selected

    # divergence field (positive and negative divergences in case of non-divergence-free models)
    # NOTE: We can imput a whole field because it only depends on the point input
    div_field = (torch.autograd.grad(
        outputs = mean_pred,
        inputs = x_grad,
        grad_outputs = u_indicator,
        create_graph = True
        )[0][:, 0] + torch.autograd.grad(
        outputs = mean_pred,
        inputs = x_grad,
        grad_outputs = v_indicator,
        create_graph = True
        )[0][:, 1])
    
    return div_field


def draw_n_samples_block_input(mean, covar, n_samples, max_jitter = 1e-2):
    """We draw n samples from a bivariate normal distribution with mean and full covariance using torch.

    Args:
        mean (torch.Size([N_FULL, 2])): 
            The columns preseny u and v components of the mean vector
        covar ([N_FULL * 2, N_FULL * 2]): 
            The full covariance matrix is outputted by my model. BE AWARE THAT THIS FUNCTION HANDLES INPUTS WITH FULL BLOCK STRUCTURE (Rows: u1, u2, u3 [...], v1, v2, v3 [...] vn and Columnns also u1, u2, u3 [...], v1, v2, v3 [...]) RATHER THAN INTERLEAVE BLOCK STRUCTURE (Rows: u1, v2, u2, v2 [...] un, vn). torch take this interleaved, mini-block structure as input
        n_samples (int): 
            number of samples that should be returned.
        epsilon (float):
            small value to ensure positive definiteness of covariance matrix
    """
    # Extract N_FULL
    N_ALL = mean.shape[0]
    N_SIDE = int(np.sqrt(N_ALL))

    ### Reshape covar ###
    # extract all 4 blocks (big blocks)
    covar_uu = covar[:N_ALL, :N_ALL]
    covar_uv = covar[:N_ALL, N_ALL:]
    covar_vu = covar[N_ALL:, :N_ALL]
    covar_vv = covar[N_ALL:, N_ALL:]

    mini_blocks = torch.stack([
            torch.stack([covar_uu, covar_uv], dim = -1), # same as dim = 2, torch.Size([400, 400, 2])
            torch.stack([covar_vu, covar_vv], dim = -1) # torch.Size([400, 400, 2])
        ], dim = -1) # same as dim = 3, torch.Size([400, 400, 2, 2])
    
    # for N_ALL = 400
    mini_blocks = mini_blocks.permute(0, 2, 1, 3)  # [400, 2, 400, 2]

    # Combine first 2 dims into one and last two dims into one
    covar_interleave = mini_blocks.reshape(N_ALL * 2, N_ALL * 2)

    ### SAMPLE ###
    mean_flat = mean.reshape(-1) # reshape goes row-wise so we have [u1, v1, u2, v2, ...]

    # This alters the  approach a fair bit Ensures this matrix symmetric and positive definite
    # covar_psd = covar_interleave @ covar_interleave.T  # Ensures this matrix symmetric and positive definite

    # Make symmetric
    covar_symmetric = 0.5 * (covar_interleave + covar_interleave.T)

    # Determine jitter magnitude needed to make the covariance matrix positive definite
    # min_eigval = torch.linalg.eigvalsh(covar_symmetric).min()
    # jitter = (- min_eigval + jitter_buffer).clamp(min = jitter_buffer)
    # covar_symmetric += torch.eye(covar_symmetric.shape[0]) * jitter

    eye = torch.eye(covar_symmetric.shape[0], device = covar_symmetric.device)
    jitter = 1e-6

    # Add as much jitter to covariance matrix as needed to make it positive definite
    while jitter <= max_jitter:
        try:
            # Try Cholesky
            torch.linalg.cholesky(covar_symmetric + jitter * eye)

            # Success: return adjusted matrix
            covar_symmetric = covar_symmetric + jitter * eye
            print(f"Jitter: {jitter}")
            break  # <--- Exit the loop once PD is achieved

        except RuntimeError:
            jitter *= 10  # Increase jitter e.g. 1e-6 > 1e-5 > 1e-4 ...
    
    else:
        warnings.warn("Failed to make matrix positive definite. Trying work around")
        emergency_jitter = 1e-6
        covar_symmetric = (covar_symmetric @ covar_symmetric.T)  # Ensures this matrix symmetric and positive definite
        # Loop again
        while emergency_jitter <= max_jitter:
            try:
                # Try Cholesky
                torch.linalg.cholesky(covar_symmetric + emergency_jitter * eye)

                # Success: return adjusted matrix
                covar_symmetric = covar_symmetric + emergency_jitter * eye
                print(f"Emergency Jitter: {emergency_jitter}")
                break  # <--- Exit the loop once PD is achieved

            except RuntimeError:
                emergency_jitter *= 10  # Increase jitter e.g. 1e-6 > 1e-5 > 1e-4 ...

    # Empty container for samples
    samples = torch.empty((n_samples, N_SIDE, N_SIDE, 2))

    # Distribution in torch
    mvn = torch.distributions.MultivariateNormal(loc = mean_flat, covariance_matrix = covar_symmetric)
    
    for i in range(n_samples):
        sample = mvn.sample().reshape(N_SIDE, N_SIDE, 2).unsqueeze(0)
        samples[i] = sample

    return samples
