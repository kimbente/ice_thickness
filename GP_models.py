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
import torch.optim as optim
from torch.func import vmap

import numpy as np
import math

from kernels import *
from simulate import *

def GP_predict(
        x_train,
        y_train,
        x_test, 
        hyperparameters,
        mean_func = None,
        divergence_free_bool = True):
    """ 
    Predicts the mean and covariance of the test data given the training data and hyperparameters

    Args:
        x_train (torch.Size([n_train, 2])): x1 and x2 coordinates
        y_train (torch.Size([n_train, 2])): u and v, might be noisy
        x_test (torch.Size([n_test, 2])): x1 and x2 coordinates
        hyperparameters (list): varying length depending on kernel
        mean_func (function, optional): mean function. Defaults to None. Inputs torch.Size([n_test, 2]) and returns torch.Size([n_test, 2] too
        divergence_free_bool (bool, optional): _description_. Defaults to True.

    Returns:
        predictive_mean (torch.Size([n_test, 2])):
        predictive_covariance (torch.Size([n_test, n_test])):
        lml (torch.Size([1])): (positive) log marginal likelihood
    """
    # Extract first hyperparameter sigma_n: noise - given, not optimised
    sigma_n = hyperparameters[0]
    
    # Extract number of rows (data points) in x_test
    n_test = x_test.shape[0]
    
    kernel_func = divergence_free_se_kernel if divergence_free_bool else block_diagonal_se_kernel

    # default is zero mean
    if mean_func == None:
        mean_y_train = torch.zeros_like(x_train)
        mean_y_test = torch.zeros_like(x_test)
    else:
        # apply model with vmap
        mean_y_train = vmap(mean_func)(x_train)
        mean_y_test = vmap(mean_func)(x_test)

    # Outputs kernel of shape torch.Size([2 * n_train, 2 * n_train])
    K_train_train = kernel_func(
        x_train, 
        x_train, 
        hyperparameters)

    # Add noise to the diagonal
    K_train_train_noisy = K_train_train + torch.eye(K_train_train.shape[0], device = K_train_train.device) * sigma_n**2

    # torch.Size([2 * n_train, 2 * n_test])
    # K_* in Rasmussen is (x_train, X_test)
    K_train_test = kernel_func(
        x_train, 
        x_test,
        hyperparameters)

    # matrix transpose
    # torch.Size([2 * n_test, 2 * n_train])
    K_test_train = K_train_test.mT

    K_test_test = kernel_func(
        x_test,
        x_test,
        hyperparameters)
    
    # Determine L - torch.Size([2 * n_train, 2 * n_train])
    L = torch.linalg.cholesky(K_train_train_noisy, upper = False)
    # L.T \ (L \ y) in one step - torch.Size([2 * n_train, 1])

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

    # matmul: torch.Size([1, 10]) * torch.Size([10, 1])
    # 0.5 * y^T * alpha
    # squeeze to remove redundant dimension
    # y_train are noisy data observations
    lml_term1 = - 0.5 * torch.matmul(y_train_minus_mean_flat.T, alpha).squeeze()

    # sum(log(L_ii)))
    lml_term2 = - torch.sum(torch.log(torch.diagonal(L)))

    # Constant term - technically not optimised 
    # n/2 * log(2 * pi)
    lml_term3 = - (y_train.shape[0]/2) * torch.log(torch.tensor(2 * math.pi))

    lml = lml_term1 + lml_term2 + lml_term3

    return predictive_mean, predictive_covariance, lml

def predict(X_train,
            Y_train_noisy,
            X_test, 
            hyperparameters,
            divergence_free_bool = True):
    """ 
    Predicts the mean and covariance of the test data given the training data and hyperparameters

    Args:
        X_train (torch.Size([n_train, 2])): x1 and x2 coordinates
        Y_train_noisy (torch.Size([n_train, 2])): u and v
        X_test (torch.Size([n_test, 2])): x1 and x2 coordinates
        hyperparameters (list): varying length depending on kernel
        divergence_free_bool (bool, optional): _description_. Defaults to True.

    Returns:
        predictive_mean (torch.Size([n_test, 2])):
        predictive_covariance (torch.Size([n_test, n_test])):
        lml (torch.Size([1])): (positive) log marginal likelihood
    """
    
    # Extract first hyperparameter sigma_n: noise - given, not optimised
    sigma_n = hyperparameters[0]
    
    # Extract number of rows (data points) in X_train and X_test
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]
    
    kernel_func = divergence_free_se_kernel if divergence_free_bool else block_diagonal_se_kernel

    # Outputs kernel of shape torch.Size([2 * n_train, 2 * n_train])
    K_train_train = kernel_func(
        X_train, 
        X_train, 
        hyperparameters)

    # Add noise to the diagonal
    K_train_train_noisy = K_train_train + torch.eye(K_train_train.shape[0], device = K_train_train.device) * sigma_n**2

    # torch.Size([2 * n_train, 2 * n_test])
    # K_* in Rasmussen is (X_train, X_test)
    K_train_test = kernel_func(
        X_train, 
        X_test,
        hyperparameters)

    # matrix transpose
    # torch.Size([2 * n_test, 2 * n_train])
    K_test_train = K_train_test.mT

    K_test_test = kernel_func(
        X_test,
        X_test,
        hyperparameters)
    
    # Determine L - torch.Size([2 * n_train, 2 * n_train])
    L = torch.linalg.cholesky(K_train_train_noisy, upper = False)
    # L.T \ (L \ y) in one step - torch.Size([2 * n_train, 1])

    # Make y flat by concatenating u and v (both columns) AFTER each other
    # torch.Size([2 x n_train, 1])
    Y_train_noisy_flat = torch.cat([Y_train_noisy[:, 0], Y_train_noisy[:, 1]]).unsqueeze(-1)

    # alpha: torch.Size([2 x n_train, 1])
    alpha = torch.cholesky_solve(Y_train_noisy_flat, L, upper = False)

    # matrix multiplication
    # torch.Size([2 * n_test, 2 * n_train]) * torch.Size([2 * n_train, 1])
    # alpha needs to be changed to datatype double because K is
    # predictive mean now is torch.Size([2 * n_test]) (after squeezing explicit last dimension)
    predictive_mean = torch.matmul(K_test_train, alpha.double()).squeeze()
    # reshape to separate u and v (which were concatnated after each other)
    predictive_mean = torch.cat([predictive_mean[:n_test].unsqueeze(-1), predictive_mean[n_test:].unsqueeze(-1)], dim = -1)

    # Step 3: Solve for V = L^-1 * K(X_*, X)
    # K_* is K_train_test
    # L is lower triangular
    v = torch.linalg.solve_triangular(L, K_train_test, upper = False)
    # same as
    # v = torch.linalg.solve(L, K_train_test)
    # torch.matmul(v, v.T) would give the wrong shape
    predictive_covariance = K_test_test - torch.matmul(v.T, v)

    # matmul: torch.Size([1, 10]) * torch.Size([10, 1])
    # 0.5 * y^T * alpha
    # squeeze to remove redundant dimension
    # Y_train_noisy are noisy data observations
    lml_term1 = - 0.5 * torch.matmul(Y_train_noisy_flat.T, alpha).squeeze()

    # sum(log(L_ii)))
    lml_term2 = - torch.sum(torch.log(torch.diagonal(L)))

    # Constant term - technically not optimised 
    # n/2 * log(2 * pi)
    lml_term3 = - (Y_train_noisy.shape[0]/2) * torch.log(torch.tensor(2 * math.pi))

    lml = lml_term1 + lml_term2 + lml_term3

    return predictive_mean, predictive_covariance, lml
 

from torch import optim

### OPTIMISATION ###
def optimise_hypers_on_train(
        hyperparameters_initial, 
        X_train, 
        Y_train_noisy, 
        X_test,
        divergence_free_bool, 
        max_optimisation_iterations = 2000,
        patience = 20,
        learning_rate = 0.001):

        # Clone hyperparameters to avoid modifying the original tensor
        # preserve what requires grad
        hyperparameters = [h.clone().detach().requires_grad_(h.requires_grad) for h in hyperparameters_initial]

        lml_log = []

        _, _, lml_initial = predict(
                X_train,
                Y_train_noisy,
                X_test,
                hyperparameters,
                divergence_free_bool) # Pass in the divergence free boolean
        
        lml_log.append(lml_initial.item())
        
        # print(f"Initial hyperparameters: {', '.join(f'{h.detach().numpy()[0]:.3f}' for h in hyperparameters)}")
        formatted_hypers = []
        for h in hyperparameters:
                h_np = h.detach().numpy()
                if h_np.ndim == 0:  # Scalar tensor
                        formatted_hypers.append(f"{h_np:.3f}")
                elif h_np.ndim == 1 and h_np.shape[0] == 1:  # Single-element 1D tensor
                        formatted_hypers.append(f"{h_np[0]:.3f}")
                else:  # Higher-dimensional tensors (e.g., matrices)
                        formatted_hypers.append(str(h_np))  # Convert to string for safe printing

        print(f"Initial hyperparameters: {', '.join(formatted_hypers)}")

        print(f"Initial LML (higher is better): {lml_initial.item():.2f}")
        print()
        
        # It doesn't hurt to pass in ones here that do NOT require grad
        optimizer = optim.Adam(
                hyperparameters, 
                lr = learning_rate) 

        best_loss = float('inf') # initialse as infinity
        best_hypers = None
        no_improvement_count = 0

        for trial in range(max_optimisation_iterations):
                
                # Compute nlml
                _, _, lml = predict(
                X_train,
                Y_train_noisy,
                X_test,
                hyperparameters,
                divergence_free_bool)

                lml_log.append(lml.item())
                
                # We are minimising the negative log marginal likelihood, like a loss function
                loss = - lml # NLML

                # Check for improvement
                if loss < best_loss:
                        best_loss = loss.item() # If better than current, save loss and hypers
                        # we need to clone and not reference
                        best_hypers = [h.clone().detach() for h in hyperparameters]
                
                        no_improvement_count = 0  # Reset counter

                else:
                        no_improvement_count += 1  # Increase counter

                ### CASE 1: Early stopping
                if no_improvement_count >= patience:
                # Printing current state
                        print(f"The optimisation processes is stopped early, after {trial+1}/{max_optimisation_iterations} iterations, due to loss stagnation.")

                        formatted_hypers = []
                        for h in best_hypers:
                                h_np = h.detach().numpy()
                                if h_np.ndim == 0:  # Scalar tensor
                                        formatted_hypers.append(f"{h_np:.3f}")
                                elif h_np.ndim == 1 and h_np.shape[0] == 1:  # Single-element 1D tensor
                                        formatted_hypers.append(f"{h_np[0]:.3f}")
                                else:  # Higher-dimensional tensors (e.g., matrices)
                                        formatted_hypers.append(str(h_np))  # Convert to string for safe printing

                        print(f"Best hyperparameters: {', '.join(formatted_hypers)}")

                        # print(f"Best hyperparameters: {', '.join(f'{h.detach().numpy().item():.3f}' for h in best_hypers)}")
                        print(f"Best LML (higher is better): {(- best_loss):.2f}")
                        
                        break
                
                optimizer.zero_grad()  # Reset gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update hypers

                # if trial % 10 == 0:  # Print every 10 iterations
                #        print(f"Current hyperparameters: {', '.join(f'{h.detach().numpy()[0]:.3f}' for h in hyperparameters)}")
                #        print(f"Current NLML: {nlml.item():.2f}")

        ### CASE 2: Regular stopping after iterations are up
        print(f"Optimisation complete after {trial+1}/{max_optimisation_iterations} iterations. Maybe consider adjusting the optimisation scheme (e.g. learning rate, max iterations, patience, etc.).")

        formatted_hypers = []
        for h in best_hypers:
                h_np = h.detach().numpy()
                if h_np.ndim == 0:  # Scalar tensor
                        formatted_hypers.append(f"{h_np:.3f}")
                elif h_np.ndim == 1 and h_np.shape[0] == 1:  # Single-element 1D tensor
                        formatted_hypers.append(f"{h_np[0]:.3f}")
                else:  # Higher-dimensional tensors (e.g., matrices)
                        formatted_hypers.append(str(h_np))  # Convert to string for safe printing

        print(f"Best hyperparameters: {', '.join(formatted_hypers)}")
        print(f"Best LML (higher is better): {(- best_loss):.2f}") # The loss is NLML so LML is - loss
        
        return best_hypers, lml_log
