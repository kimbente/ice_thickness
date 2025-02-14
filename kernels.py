import torch
import numpy as np

def divergence_free_se_kernel(
        row_tensor, # torch.Size([n_rows, 2])
        column_tensor, # torch.Size([n_columns, 2])
        hyperparameters):
    
    """
    Calculate the divergence-free SE kernel for two sets of points in 2D space.
    R^2 -> R^2

    Inputs:
        row_tensor: torch.Size([n_rows, 2])
        column_tensor: torch.Size([n_columns, 2])
        hyperparameters: list of length 2 containing sigma_f and l

    Returns:
        K: torch.Size([n_rows * 2, n_columns * 2])
    """
    
    # We calculate the kernel for each pair of points
    
    # Extract hyperparameters
    sigma_f = hyperparameters[0]
    l = hyperparameters[1]

    # Add dimension (broadcasting) for difference calculation
    # torch.Size([n_rows, 1, 2]) - 1 is for n_columns
    row_tensor_expanded = row_tensor[:, None, :]
    # torch.Size([1, n_columns, 2]) - 1 is for n_rowns
    column_tensor_expanded = column_tensor[None, :, :]

    # [:, :, 0] are the x1 differences and [:, :, 1] are the x2 differences
    # yields negative values as well
    diff = row_tensor_expanded - column_tensor_expanded

    ### 2x2 BLOCKS ###
    # x2 diffs: torch.Size([n_rows, n_columns])
    upper_left = (1 - diff[:, :, 1].square().div(l.square())).div(l.square())

    # x1 diffs: torch.Size([n_rows, n_columns])
    lower_right = (1 - diff[:, :, 0].square().div(l.square())).div(l.square())

    # Elementwise multiplication of x1 and x2 diffs and division by scalar
    # Matlab version has negative values here!
    upper_right = torch.prod(diff, dim = -1).div(l**4)

    # same as other off-diagonal block
    lower_left = upper_right

    # x1 diffs: torch.Size([n_rows, n_columns])
    lower_right = (1 - diff[:, :, 0].square().div(l.square())).div(l.square())

    # Concatenate upper and lower blocks column-wise, and then concatenate them row-wise
    # torch.Size([2 * n_train, 2 * n_test])
    blocks = torch.cat((
        torch.cat((upper_left, upper_right), 1), 
        torch.cat((lower_left, lower_right), 1)
        ), 0)

    # torch.Size([2 * n_row, 2 * n_column])
    # elementwise multiplication
    # sum squared difference over x1 and x2, divide by -2 * l^2, and exponentiate. Tile for blocks
    K = sigma_f.square() * blocks.mul(diff.square().sum(dim = -1).div(-2 * l.square()).exp().tile(2, 2))

    return K

