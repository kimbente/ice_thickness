import torch

def compute_RMSE(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.square(y_true - y_pred)))

def compute_MAE(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def compute_NLL(y_true, y_mean_pred, y_covar_pred):
    """NLL quantifies how well the predicted Gaussian distribution fits the observed data.
    Sparse format: each of the N points has its own 2×2 covariance matrix. (This is more than just the diagonal of the covariance matrix, but not the full covar.)

    Args:
        y_true (torch.Size([N, 2])): true, observed vectors
        y_mean_pred (torch.Size([N, 2])): mean predictions
        y_covar_pred (torch.Size([N x 2, N x 2])): predicted covariance matrix
            if N = 400, then y_covar_pred is torch.Size([800, 800]) so 640000 elements
            N x 2 x 2 = only 1600 elements
    """
    # Extract number of points/cardinality
    N = y_true.shape[0]

    ### STEP 1:
    # We reshape y_covar_pred, the predictive covariance matrix, to a sparse format
    # Change format of y_covar_pred from (N x 2, N x 2) to (N, 2, 2) so N (2, 2) matrices:
    
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

    # batch tensor
    # At each point N, what is the predicted variance of y1 and y2 and what is the predicted covariance between y1 and y2? (symmetric)
    covar_N22 = torch.cat([col1.unsqueeze(-1), col2.unsqueeze(-1)], dim = -1)
    
    ### STEP 2:
    # Calculate the log likelihood in torch

    # torch.Size([N, 2]) differences in y
    diff = y_mean_pred - y_true

    # batch matrix multiplication (n, 2) * (n, 2, 2) -> (n, 2)
    # torch.inverse(covar_N22): Inverts each (2×2) covariance matrix
    # diff.unsqueeze(-1): Reshapes (N, 2) → (N, 2, 1) for matrix multiplication
    # .sum(dim=-1): Sum across both dimensions (y1, y2).
    # the inverse is batchwise
    mahalanobis_dist = torch.mul(torch.matmul(torch.inverse(covar_N22), diff.unsqueeze(-1)).squeeze(-1), diff).sum(dim = -1)

    # element-wise determenant of all 2x2 matrices
    log_det_Sigma = torch.logdet(covar_N22)

    # element-wise log likelihood
    # Gaussian log-likelihood formula: 2D
    log_likelihood_tensor = -0.5 * (mahalanobis_dist + log_det_Sigma + 2 * torch.log(torch.tensor(2 * torch.pi)))

    # sum over all N (because we are in the log domain)
    nll = - log_likelihood_tensor.sum()

    return nll

#### Full version
def compute_NLL_full(y_true, y_mean_pred, y_covar_pred, jitter = 1e-3):
    """Computes Negative Log-Likelihood (NLL) using the full covariance matrix.

    Args:
        y_true (torch.Tensor): True observations of shape (N, 2).
        y_mean_pred (torch.Tensor): Mean predictions of shape (N, 2).
        y_covar_pred (torch.Tensor): Full predicted covariance matrix of shape (N*2, N*2).(BLOCK FORMAT)
        jitter (float, optional): Small value added to the diagonal for numerical stability. Defaults to 1e-3 - quite high.

    Returns:
        torch.Tensor: Negative Log-Likelihood (NLL) scalar.
    """
    # Number of points
    N = y_true.shape[0]
    
    # Flatten y_true and y_mean_pred to match covariance matrix shape
    # Reshape makes it [u1, v1, u2, v2, u3, v3, ...] instead of [u1, u2, u3, ..., v1, v2, v3, ...]
    y_true_flat = y_true.reshape(-1, 1)  # Shape: (N*2, 1)
    y_mean_pred_flat = y_mean_pred.reshape(-1, 1)  # Shape: (N*2, 1)

    ### STEP 1: Stabilize covariance matrix
    eps = torch.eye(y_covar_pred.shape[0], device = y_covar_pred.device) * jitter
    y_covar_pred_stable = y_covar_pred + eps  # Regularization to ensure invertibility

    ### STEP 2: Compute Mahalanobis distance efficiently
    diff = y_mean_pred_flat - y_true_flat  # Shape: (N*2, 1)
    
    # Solve Σ⁻¹ (y - μ) using Cholesky decomposition for better stability
    chol = torch.linalg.cholesky(y_covar_pred_stable)
    mahalanobis_dist = torch.cholesky_solve(diff, chol).T @ diff
    mahalanobis_dist = mahalanobis_dist.squeeze()

    ### STEP 3: Compute log-determinant robustly
    sign, log_det_Sigma = torch.linalg.slogdet(y_covar_pred_stable)
    
    # If the determinant is non-positive, return a large NLL to indicate instability
    if sign <= 0:
        return torch.tensor(float("inf"), device = y_true.device)
    
    ### STEP 4: Compute negative log-likelihood (NLL)
    d = N * 2  # Dimensionality (since we have two outputs per point)
    log_likelihood = -0.5 * (mahalanobis_dist + log_det_Sigma + d * torch.log(torch.tensor(2 * torch.pi, device = y_true.device)))

    return - log_likelihood  # Negative log-likelihood

# This was replaced
    """def log_likelihood_test(predictive_mean, predictive_covar, Y_test):

    ### STEP 1 ###
    # Change format of predctive covar from (N x 2, N x 2) to (N, 2, 2) so N (2, 2) matrices:
    n_test = Y_test.shape[0]

    var_y1_y1 = torch.diag(predictive_covar[:n_test, :n_test])
    var_y2_y2 = torch.diag(predictive_covar[n_test:, n_test:])

    covar_y1_y2 = torch.diag(predictive_covar[:n_test, n_test:])
    covar_y2_y1 = torch.diag(predictive_covar[n_test:, :n_test])

    col1 = torch.cat([var_y1_y1.unsqueeze(-1), covar_y1_y2.unsqueeze(-1)], dim = -1)
    col2 = torch.cat([covar_y2_y1.unsqueeze(-1), var_y2_y2.unsqueeze(-1)], dim = -1)
    
    covar_N22 = torch.cat([col1.unsqueeze(-1), col2.unsqueeze(-1)], dim = -1)

    ### STEP 2 ###
    # cal log likelihood in torch 
    # torch.Size([N, 2]) differences in y
    diff = predictive_mean - Y_test

    # batch matrix multiplication (n, 2) * (n, 2, 2) -> (n, 2)
    mahalanobis_dist = torch.mul(torch.matmul(torch.inverse(covar_N22), diff.unsqueeze(-1)).squeeze(-1), diff).sum(dim = -1)
    
    # element-wise determenant of all 2x2 matrices
    log_det_Sigma = torch.logdet(covar_N22)

    # element-wise log likelihood
    log_likelihood_tensor = -0.5 * (mahalanobis_dist + log_det_Sigma + 2 * torch.log(torch.tensor(2 * torch.pi)))

    # sum over all N (because we are in the log domain)
    log_likelihood = log_likelihood_tensor.sum()
    
    # scalar
    return log_likelihood
    """