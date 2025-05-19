import torch

##############################
### Root Mean Square Error ###
##############################

def compute_RMSE(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.square(y_true - y_pred)))

############################
### Mean Absolute Error ###
############################

def compute_MAE(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

##########################
### NLL sparse version ###
##########################

def compute_NLL_sparse(y_true, y_mean_pred, y_covar_pred):
    """ Computes a sparse version of the Negative Log-Likelihood (NLL) for a 2D Gaussian distribution. This sparse version neglects cross-covariance terms and is more efficient for large datasets.
    
    NLL: The NLL quantifies how well the predicted Gaussian distribution fits the observed data.
    Sparse format: each of the N points has its own 2×2 covariance matrix. (This is more than just the diagonal of the covariance matrix, but not the full covar.)

    Args:
        y_true (torch.Tensor): True observations of shape (N, 2).
        y_mean_pred (torch.Tensor): Mean predictions of shape (N, 2).
        y_covar_pred (torch.Tensor): Full predicted covariance matrix of shape (N * 2, N * 2).(BLOCK FORMAT) [u1, u2, u3, ..., v1, v2, v3, ...]
            If N = 400, then y_covar_pred is torch.Size([800, 800]) so 640000 elements N x 2 x 2 = only 1600 elements.
        jitter (float, optional): Small value added to the diagonal for numerical stability. Defaults to 0.5 * 1e-2 - quite high but we need to keep it consistent across all models.

    Returns:
        torch.Tensor(): Negative Log-Likelihood (NLL) scalar.
    """
    # Extract number of points
    N = y_true.shape[0]

    # Step 1: Sparsify the covariance matrix
    # Change format of y_covar_pred from (N x 2, N x 2) to (N, 2, 2) so N (2, 2) matrices.
    # NOTE: This is a sparse version of the covariance matrix, neglecting cross-covariance terms.

    # extract diagonal of upper left quadrant: variance of the first output (y1) at each point.
    var_y1_y1 = torch.diag(y_covar_pred[:N, :N])
    # extract diagonal of ulower right quadrant: variance of the second output (y2) at each point
    var_y2_y2 = torch.diag(y_covar_pred[N:, N:])

    # extract diagonal of upper right quadrant: How much do y1 and y2 covary at this point
    covar_y1_y2 = torch.diag(y_covar_pred[:N, N:])
    # extract diagonal of lower left quadrant
    covar_y2_y1 = torch.diag(y_covar_pred[N:, :N])

    col1 = torch.cat([var_y1_y1.unsqueeze(-1), covar_y1_y2.unsqueeze(-1)], dim = -1)
    col2 = torch.cat([covar_y2_y1.unsqueeze(-1), var_y2_y2.unsqueeze(-1)], dim = -1)

    # At each point N, what is the predicted variance of y1 and y2 and 
    # what is the predicted covariance between y1 and y2? (symmetric)
    covar_N22 = torch.cat([col1.unsqueeze(-1), col2.unsqueeze(-1)], dim = -1) # shape: torch.Size([N, 2, 2])


    # STEP 2: Compute Mahalanobis distance efficiently
    # Compute the difference between the true and predicted values (y - μ)
    # NOTE: order is (true - pred) to match the Mahalanobis distance formula
    # NOTE: we can also keep this shape
    diff = y_true - y_mean_pred   # Shape: (N, 2)
    
    # Reshape diff to (N, 2, 1) to do matrix multiplication with (N, 2, 2)
    diff = diff.unsqueeze(-1)  # shape: (N, 2, 1)

    sigma_inverse = torch.inverse(covar_N22) # shape: torch.Size([N, 2, 2])

    # Compute (Σ⁻¹ @ diff) → shape: (N, 2, 1)
    maha_component = torch.matmul(sigma_inverse, diff)

    # Compute (diff^T @ Σ⁻¹ @ diff) for each point → shape: (N, 1, 1)
    # transpose diff to (N, 1, 2) for matrix multiplication
    mahalanobis_distances = torch.matmul(diff.transpose(1, 2), maha_component)

    # Sum (N, ) distances to get a single value
    mahalanobis_distances = mahalanobis_distances.squeeze().sum()


    # STEP 3: Log determinant of the covariance matrix

    # element-wise determinant of all 2x2 matrices: sum
    sign, log_absdet = torch.slogdet(covar_N22)
    if not torch.all(sign > 0):
        print("Warning: Non-positive definite matrix encountered.")
        return torch.tensor(float("inf"), device = covar_N22.device)
    log_det_Sigma = log_absdet.sum()


    # STEP 4: Compute normalisation term
    d = N * 2  # Dimensionality (since we have two outputs per point)
    normalisation_term = d * torch.log(torch.tensor(2 * torch.pi, device = y_true.device))

    # Step 5: Combine 3 scalars into negative log-likelihood (NLL)
    # Gaussian log-likelihood formula: 2D
    # NOTE: Gaussian log-likelihood 2D formula
    log_likelihood =  - 0.5 * (mahalanobis_distances + log_det_Sigma + normalisation_term)

    # return the negative log-likelihood
    return - log_likelihood

########################
### NLL Full version ###
########################

def compute_NLL_full(y_true, y_mean_pred, y_covar_pred, jitter = 0.5 * 1e-2):
    """Computes Negative Log-Likelihood (NLL) using the full covariance matrix.

    Args:
        y_true (torch.Tensor): True observations of shape (N, 2).
        y_mean_pred (torch.Tensor): Mean predictions of shape (N, 2).
        y_covar_pred (torch.Tensor): Full predicted covariance matrix of shape (N*2, N*2).(BLOCK FORMAT) [u1, u2, u3, ..., v1, v2, v3, ...]
        jitter (float, optional): Small value added to the diagonal for numerical stability. Defaults to 0.5 * 1e-2 - quite high but we need to keep it consistent across all models.

    Returns:
        torch.Tensor(): Negative Log-Likelihood (NLL) scalar.
    """
    # Extract number of points
    N = y_true.shape[0]
    
    # STEP 1: Compute Mahalanobis distance efficiently

    # NOTE: Flatten y_true and y_mean_pred & match covariance matrix shape (BLOCK structure)
    y_true_flat = torch.concat([y_true[:, 0], y_true[:, 1]], dim = 0).unsqueeze(-1)  # Shape: (2 * N, 1)
    y_mean_pred_flat = torch.concat([y_mean_pred[:, 0], y_mean_pred[:, 1]], dim = 0).unsqueeze(-1)  # Shape: (2 * N, 1)

    # Compute the difference between the true and predicted values (y - μ)
    # NOTE: order is (true - pred) to match the Mahalanobis distance formula
    diff = y_true_flat - y_mean_pred_flat   # Shape: (2 * N, 1)

    # STEP 2: Stabilize covariance matrix with fixed jitter to ensure torch.linalg.cholesky() works
    # NOTE: as this is our key metric it is better to add the same jitter to all elements. Thus rather than a loop, we add a fixed small value to the diagonal
    y_covar_pred_stable = y_covar_pred + torch.eye(y_covar_pred.shape[0], device = y_covar_pred.device) * jitter
    
    # Solve Σ⁻¹ using Cholesky decomposition to get lower-triangular matrix L for LL^T = Σ
    chol = torch.linalg.cholesky(y_covar_pred_stable) # Shape: (2 * N, 2 * N)

    # Solve (y - μ)T Σ⁻¹ (y - μ) using Cholesky decomposition for better stability
    mahalanobis_dist = (torch.cholesky_solve(diff, chol).T @ diff).squeeze() # Shape: (1,)

    # STEP 3: Compute log-determinant robustly
    # HACK: avoids underflow/overflow
    sign, log_det_Sigma = torch.linalg.slogdet(y_covar_pred_stable) # sign = 1 or -1, log_det_Sigma = log determinant (scalar)
    
    # If the determinant is non-positive, return a large NLL to indicate instability
    if sign <= 0:
        print("Warning: Non-positive determinant encountered. Returning large NLL.")
        return torch.tensor(float("inf"), device = y_true.device)
    
    # STEP 4: Compute normalisation term
    d = N * 2  # Dimensionality (since we have two outputs per point)
    normalisation_term = d * torch.log(torch.tensor(2 * torch.pi, device = y_true.device))

    # Step 5: Combine 3 terms into negative log-likelihood (NLL)
    log_likelihood = -0.5 * (mahalanobis_dist + log_det_Sigma + normalisation_term)

    return - log_likelihood  # Negative log-likelihood

########################
### Divergence field ###
########################

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