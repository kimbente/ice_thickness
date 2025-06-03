############
### NLML ###
############

# In this version we use the precomputed alpha and L matrices to compute the LML to save compute
# Negative Log Marginal Likelihood (NLML, i.e. - LML) is the objective function that we want to minimize
# equivalent to maximising the lml
# for HYPERPARAMETER OPTIMIZATION

# Rasmussen and Williams 2006 Algorithm 2.1, line 8, page 37
# Note: This is indeed the lml (log marginal likelihood) and not the nlml (negative log marginal likelihood)
# see https://gregorygundersen.com/blog/2019/09/12/practical-gp-regression/ for derivations

# TERM 1: Model-data fit
# 0.5 * y^T * alpha
# where y^T is a row vector of shape [1, n] and alpha is a column vector of shape [n, 1] so that the product is a scalar

# TERM 2: Complexity
# sum(log(L_ii)))
# where L is the lower triangular matrix of the Cholesky decomposition of the covariance matrix K
# we sum the diagonal elements of L after taking the log

# TERM 3: Normalisation term
# n/2 * log(2*pi)
# where n is the number of data points
# this is a constant term

import torch

from kernels import *
from simulate import *

###############################
### INTERWOVEN structure GP ###
###############################
# NOTE: In the experiment the interwoven structure is more stable during training

# try interleaved variant
def GP_predict(
    x_train,
    y_train,
    x_test, 
    hyperparameters,
    mean_func = None,
    divergence_free_bool = True, 
    return_L = False):
    """ 
    Predicts the mean and covariance of the test data given the training data and hyperparameters (or fixed noise inputs). This implementation uses the interleaved structure.

    Args:
        x_train (torch.Size([n_train, 2])): x1 and x2 coordinates
        y_train (torch.Size([n_train, 2])): u and v, might be noisy
        x_test (torch.Size([n_test, 2])): x1 and x2 coordinates

        hyperparameters (list):    
            varying length depending on kernel
            [sigma_n, sigma_f, l]: 
                sigma_n can either be a torch.Size([1]) or a torch.Size([y_train.shape[0]])

        mean_func (function, optional): 
            mean function. Defaults to None. Inputs torch.Size([n_test, 2]) and returns torch.Size([n_test, 2] too.

        divergence_free_bool (bool, optional): Indicating whether we use a df kernel or a regular kernel. Defaults to True.

    Returns:
        predictive_mean (torch.Size([n_test, 2])):
        predictive_covariance (torch.Size([n_test, n_test])):
        lml (torch.Size([1])): (positive) log marginal likelihood of (x_train, y_train)
        optional: 
        L
    """
    # Extract the first hyperparameter from the list (the noise level) - this can be a tensor of size [1] or [n_train] which is the vector diagonal
    sigma_n = hyperparameters[0]

    # TODO: check if we need to enforce positivity for the experiments where sigma_n is treated as hyperparameter
    # sigma_n = torch.clip(sigma_n, min = 2e-6)
    
    # Extract number of rows (data points) in x_test
    n_test = x_test.shape[0]
    
    # Select kernel function
    kernel_func = divergence_free_se_kernel if divergence_free_bool else block_diagonal_se_kernel

    ### MEAN FUNCTION ###
    # the default is a zero mean function
    if mean_func == None:
        mean_y_train = torch.zeros_like(x_train)
        mean_y_test = torch.zeros_like(x_test) # torch.Size([n_test, 2]
    else:
        mean_y_train = mean_func(x_train)
        mean_y_test = mean_func(x_test) # torch.Size([n_test, 2]

    # Outputs kernel of shape torch.Size([2 * n_train, 2 * n_train])
    K_train_train = kernel_func(
        x_train, 
        x_train, 
        hyperparameters)

    # Add noise to the diagonal (variance) of the training data
    # NOTE: This should works for both a scalar and a vector with locally varying noise levels. std gets squared.
    K_train_train_noisy = K_train_train + torch.eye(K_train_train.shape[0], device = K_train_train.device) * sigma_n**2

    # reshape after adding noise (block form)
    K_train_train_noisy_interleaved = reformat_block_kernel_to_interleaved_kernel(K_train_train_noisy)

    # Symmetry check
    if (K_train_train_noisy_interleaved != K_train_train_noisy_interleaved.mT).any():
        print("K_train_train_noisy is not symmetric")

    # K_* in Rasmussen is K(x_train, x_test)
    K_train_test = kernel_func(
        x_train, 
        x_test,
        hyperparameters) # shape: torch.Size([2 * n_test, 2 * n_train])
    
    K_train_test_interleaved = reformat_block_kernel_to_interleaved_kernel(K_train_test)

    # transpose K_train_test_interleaved to get K_test_train_interleaved
    K_test_train_interleaved = K_train_test_interleaved.mT  # shape: torch.Size([2 * n_test, 2 * n_train])

    K_test_test = kernel_func(
        x_test,
        x_test,
        hyperparameters)
    
    K_test_test_interleaved = reformat_block_kernel_to_interleaved_kernel(K_test_test)
    
    # Determine L - torch.Size([2 * n_train, 2 * n_train])
    # L.T \ (L \ y) in one step - torch.Size([2 * n_train, 1])

    ### Step 2: Cholesky decomposition with jitter
    jitter = 0.0  # Start with no jitter
    max_tries = 6
    attempt = 0

    I = torch.eye(K_train_train_noisy_interleaved.size(0), device = K_train_train_noisy_interleaved.device)

    while attempt < max_tries:
        try:
            L = torch.linalg.cholesky(K_train_train_noisy_interleaved + jitter * I)
            # if jitter > 0:
               # print(f"Cholesky succeeded with jitter = {jitter}")
            break  # Success!
        except RuntimeError:
            # if attempt == 0:
                # print("Cholesky failed without jitter. Adding jitter...")
            attempt += 1
            jitter = 1e-6 * (10 ** attempt)  # Exponential backoff
    else:
        raise RuntimeError(f"Cholesky decomposition failed after {max_tries} attempts. Final jitter: {jitter}")

    y_train_minus_mean = y_train - mean_y_train # both inputs: torch.Size([2 x n_train, 1])

    # NOTE: this is the interleaved version
    y_train_minus_mean_flat_interleaved = y_train_minus_mean.reshape(-1).unsqueeze(-1) # torch.Size([2 x n_train, 1])

    # alpha: torch.Size([2 x n_train, 1])
    alpha = torch.cholesky_solve(y_train_minus_mean_flat_interleaved, L, upper = False)

    # matrix multiplication
    # torch.Size([2 * n_test, 2 * n_train]) * torch.Size([2 * n_train, 1])
    # alpha needs to be changed to datatype double because K is
    # predictive mean now is torch.Size([2 * n_test]) (after squeezing explicit last dimension)
    predictive_mean_interleaved = torch.matmul(K_test_train_interleaved, alpha).squeeze()

    # Make mean torch.Size([n_test, 2]) and add mean_y_test which has the same format
    # NOTE: predictive mean is returned with two columns so we rename this just predictive_mean
    predictive_mean = predictive_mean_interleaved.reshape(-1, 2) + mean_y_test

    # Step 3: Solve for V = L^-1 * K(X_*, X)
    # K_* is K_train_test
    # L is lower triangular
    v = torch.linalg.solve_triangular(L, K_train_test_interleaved, upper = False)
    # same as
    # v = torch.linalg.solve(L, K_train_test)
    # torch.matmul(v, v.T) would give the wrong shape
    predictive_covariance_interleaved = K_test_test_interleaved - torch.matmul(v.T, v)

    # reshape to block form
    predictive_covariance = reformat_interleaved_kernel_to_block_kernel(predictive_covariance_interleaved)

    #################################################
    ### Log Marginal Likelihood (LML) Calculation ###
    #################################################
    # How well does the model fit the training data we have access to?

    # 0.5 * y^T * alpha
    # squeeze to remove redundant dimension
    # y_train are noisy data observations
    lml_term1 = - 0.5 * torch.matmul(y_train_minus_mean_flat_interleaved.T, alpha).squeeze()

    # Is uses only training data
    # sum(log(L_ii)))
    lml_term2 = - torch.sum(torch.log(torch.diagonal(L)))

    # Constant term - technically not optimised 
    # n/2 * log(2 * pi)
    lml_term3 = - (y_train.shape[0]/2) * torch.log(torch.tensor(2 * torch.pi))

    lml = lml_term1 + lml_term2 + lml_term3

    if return_L:
        # Return L for debugging purposes
        return predictive_mean, predictive_covariance, lml, L
    else:
        return predictive_mean, predictive_covariance, lml

#########################################################################################################
# helper functions

def reformat_block_kernel_to_interleaved_kernel(K):
    """
    Reformats any block covariance matrix (i.e. kernel) to the interleaved format. This works on square and non-square (i.e. K_train_test) matrices

    Args:
        K (torch.Size([2 * n_rows, 2 * n_columns])): Block kernel
            Shape is like: (4 big blocks for a 2D kernel)
            [[UU, UV],
            [VU, VV]]
    Returns:
        K_interleaved (torch.Size([2 * n_rows, 2 * n_columns])): Interleaved kernel
            Shape is like of both rows and columns is like [u1, v1, u2, v2, ...] where u and v (both output dims are interwoven)
    """
    # Extract dims from kernel directly
    n_rows = K.shape[0] // 2
    n_columns = K.shape[1] // 2

    # Initialise a container where we have a 2 x 2 mini covariance matrix for each pair of points
    K_interleaved = torch.zeros((n_rows, n_columns, 2, 2), device = K.device)
    
    # Fill in the interleaved kernel in 2 x 2 shape
    # e.g. the upper left block populates all upper left corners of the mini covariance matrices
    K_interleaved[:, :, 0, 0] = K[:n_rows, :n_columns]
    K_interleaved[:, :, 0, 1] = K[:n_rows, n_columns:]
    K_interleaved[:, :, 1, 0] = K[n_rows:, :n_columns]
    K_interleaved[:, :, 1, 1] = K[n_rows:, n_columns:]

    K_interleaved = K_interleaved.permute(0, 2, 1, 3).reshape(n_rows * 2, n_columns * 2)

    return K_interleaved

def reformat_interleaved_kernel_to_block_kernel(K):
    """
    Reformats any interwoven covariance matrix (i.e. kernel) to the block format. This works on square and non-square (i.e. K_train_test) matrices

    Args:
        K_interleaved (torch.Size([2 * n_rows, 2 * n_columns])): Interleaved kernel
            Shape is like of both rows and columns is like [u1, v1, u2, v2, ...] where u and v (both output dims are interwoven)
    Returns:
        K (torch.Size([2 * n_rows, 2 * n_columns])): Block kernel
            Shape is like: (4 big blocks for a 2D kernel)
            [[UU, UV],
            [VU, VV]]
    """
    # Extract dims from kernel directly
    n_rows = K.shape[0] // 2
    n_columns = K.shape[1] // 2

    # Reshape to (n_rows, n_columns, 2, 2), i.e. a mini covariance matrix for each pair of points
    K_reshaped = K.reshape(n_rows, 2, n_columns, 2).permute(0, 2, 1, 3) # shape: (n_rows, n_columns, 2, 2)

    # Now extract the 2x2 blocks into the block kernel
    K_block = torch.zeros((2 * n_rows, 2 * n_columns), device = K.device)

    K_block[:n_rows, :n_columns] = K_reshaped[:, :, 0, 0]
    K_block[:n_rows, n_columns:] = K_reshaped[:, :, 0, 1]
    K_block[n_rows:, :n_columns] = K_reshaped[:, :, 1, 0]
    K_block[n_rows:, n_columns:] = K_reshaped[:, :, 1, 1]

    return K_block


##########################
### Block structure GP ###
##########################

def GP_predict_block(
    x_train,
    y_train,
    x_test, 
    hyperparameters,
    mean_func = None,
    divergence_free_bool = True):
    """ 
    Predicts the mean and covariance of the test data given the training data and hyperparameters (or fixed noise inputs). This implementation uses the block structure.

    Args:
        x_train (torch.Size([n_train, 2])): x1 and x2 coordinates
        y_train (torch.Size([n_train, 2])): u and v, might be noisy
        x_test (torch.Size([n_test, 2])): x1 and x2 coordinates

        hyperparameters (list):    
            varying length depending on kernel
            [sigma_n, sigma_f, l]: 
                sigma_n can either be a torch.Size([1]) or a torch.Size([y_train.shape[0]])

        mean_func (function, optional): 
            mean function. Defaults to None. Inputs torch.Size([n_test, 2]) and returns torch.Size([n_test, 2] too.

        divergence_free_bool (bool, optional): Indicating whether we use a df kernel or a regular kernel. Defaults to True.

    Returns:
        predictive_mean (torch.Size([n_test, 2])):
        predictive_covariance (torch.Size([n_test, n_test])):
        lml (torch.Size([1])): (positive) log marginal likelihood of (x_train, y_train)
    """
    # Extract the first hyperparameter from the list (the noise level) - this can be a tensor of size [1] or [n_train]
    sigma_n = hyperparameters[0]

    # TODO: check if we need to enforce positivity for the experiments where sigma_n is treated as hyperparameter
    # sigma_n = torch.clip(sigma_n, min = 2e-6)
    
    # Extract number of rows (data points) in x_test
    n_test = x_test.shape[0]
    
    # Select kernel function
    kernel_func = divergence_free_se_kernel if divergence_free_bool else block_diagonal_se_kernel

    ### MEAN FUNCTION ###
    # the default is a zero mean function
    if mean_func == None:
        mean_y_train = torch.zeros_like(x_train)
        mean_y_test = torch.zeros_like(x_test)
    else:
        mean_y_train = mean_func(x_train)
        mean_y_test = mean_func(x_test)

    # Outputs kernel of shape torch.Size([2 * n_train, 2 * n_train])
    K_train_train = kernel_func(
        x_train, 
        x_train, 
        hyperparameters)

    # Add noise to the diagonal (variance) of the training data
    # NOTE: This should works for both a scalar and a vector with locally varying noise levels. std gets squared.
    K_train_train_noisy = K_train_train + torch.eye(K_train_train.shape[0], device = K_train_train.device) * sigma_n**2

    # Symmetry check
    if (K_train_train_noisy != K_train_train_noisy.mT).any():
        print("K_train_train_noisy is not symmetric")

    # K_* in Rasmussen is K(x_train, x_test)
    K_train_test = kernel_func(
        x_train, 
        x_test,
        hyperparameters) # shape: torch.Size([2 * n_test, 2 * n_train])

    # transpose K_train_test to get K_test_train
    K_test_train = K_train_test.mT  # shape: torch.Size([2 * n_test, 2 * n_train])

    K_test_test = kernel_func(
        x_test,
        x_test,
        hyperparameters)
    
    # Determine L - torch.Size([2 * n_train, 2 * n_train])
    # L = torch.linalg.cholesky(K_train_train_noisy, upper = False)
    # L.T \ (L \ y) in one step - torch.Size([2 * n_train, 1])

    ### Step 2: Cholesky decomposition with jitter
    jitter = 0.0  # Start with no jitter
    max_tries = 6
    attempt = 0
    I = torch.eye(K_train_train_noisy.size(0), device = K_train_train_noisy.device)

    while attempt < max_tries:
        try:
            L = torch.linalg.cholesky(K_train_train_noisy + jitter * I)
            if jitter > 0:
               print(f"Cholesky succeeded with jitter = {jitter}")
            break  # Success!
        except RuntimeError:
            if attempt == 0:
                print("Cholesky failed without jitter. Adding jitter...")
            attempt += 1
            jitter = 1e-6 * (10 ** attempt)  # Exponential backoff
    else:
        raise RuntimeError(f"Cholesky decomposition failed after {max_tries} attempts. Final jitter: {jitter}")

    # Make y flat by concatenating u and v (both columns) AFTER each other
    # torch.Size([2 x n_train, 1])
    # y_train_flat = torch.cat([y_train[:, 0], y_train[:, 1]]).unsqueeze(-1)
    y_train_minus_mean = y_train - mean_y_train
    y_train_minus_mean_flat = torch.cat([y_train_minus_mean[:, 0], y_train_minus_mean[:, 1]]).unsqueeze(-1)

    # alpha: torch.Size([2 x n_train, 1])
    alpha = torch.cholesky_solve(y_train_minus_mean_flat, L, upper = False)

    # matrix multiplication
    # torch.Size([2 * n_test, 2 * n_train]) * torch.Size([2 * n_train, 1])
    # alpha needs to be changed to datatype double because K is
    # predictive mean now is torch.Size([2 * n_test]) (after squeezing explicit last dimension)
    predictive_mean = torch.matmul(K_test_train, alpha).squeeze()
    # reshape to separate u and v (which were concatenated after each other)
    # ADD test mean 
    predictive_mean = torch.cat([predictive_mean[:n_test].unsqueeze(-1), predictive_mean[n_test:].unsqueeze(-1)], dim = -1) + mean_y_test

    # Step 3: Solve for V = L^-1 * K(X_*, X)
    # K_* is K_train_test
    # L is lower triangular
    v = torch.linalg.solve_triangular(L, K_train_test, upper = False)
    # same as
    # v = torch.linalg.solve(L, K_train_test)
    # torch.matmul(v, v.T) would give the wrong shape
    predictive_covariance = K_test_test - torch.matmul(v.T, v)

    #################################################
    ### Log Marginal Likelihood (LML) Calculation ###
    #################################################
    # How well does the model fit the training data we have access to?

    # 0.5 * y^T * alpha
    # squeeze to remove redundant dimension
    # y_train are noisy data observations
    lml_term1 = - 0.5 * torch.matmul(y_train_minus_mean_flat.T, alpha).squeeze()

    # Is uses only training data
    # sum(log(L_ii)))
    lml_term2 = - torch.sum(torch.log(torch.diagonal(L)))

    # Constant term - technically not optimised 
    # n/2 * log(2 * pi)
    lml_term3 = - (y_train.shape[0]/2) * torch.log(torch.tensor(2 * torch.pi))

    lml = lml_term1 + lml_term2 + lml_term3

    return predictive_mean, predictive_covariance, lml