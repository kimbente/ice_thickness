import torch

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
        hyperparameters: list of length 3 containing sigma_n, sigma_f and l

    Returns:
        K: torch.Size([n_rows * 2, n_columns * 2]) returned in block structure
    """
    
    # We calculate the kernel for each pair of points
    
    # Extract hyperparameters (except for sigma_n)
    # sigma_f_squared = torch.exp(hyperparameters[1]) # torch.exp(log_sigma_f_squared)
    # sigma_f_squared = hyperparameters[1]
    sigma_f = hyperparameters[1].to(row_tensor.device)
    l = hyperparameters[2].to(row_tensor.device)

    # HACK: ensure that lengthscale is positive (add small value to avoid division by zero)
    # with beta = 5.0 the softplus function is very close to the identity function, so that values are interpretable
    l = torch.nn.functional.softplus(l, beta = 5.0) + 1e-8

    if l.shape[0] == 1:
        lx1 = l
        lx2 = l
    else:
        lx1 = l[0]
        lx2 = l[1]

    # Add dimension (broadcasting) for difference calculation
    # torch.Size([n_rows, 1, 2]) - 1 is for n_columns
    row_tensor_expanded = row_tensor[:, None, :]
    # torch.Size([1, n_columns, 2]) - 1 is for n_rowns
    column_tensor_expanded = column_tensor[None, :, :]

    # Calculate differences for x-coordinate "features" as well as y-coordinate "features"
    # [:, :, 0] are the x1 differences and [:, :, 1] are the x2 differences
    # yields negative values as well
    diff = row_tensor_expanded - column_tensor_expanded

    ### 2x2 BLOCKS ###
    # x2 diffs: torch.Size([n_rows, n_columns])
    upper_left = (1 - diff[:, :, 1].square().div(lx2.square())).div(lx2.square())

    # x1 diffs: torch.Size([n_rows, n_columns])
    lower_right = (1 - diff[:, :, 0].square().div(lx1.square())).div(lx1.square())

    # Elementwise multiplication of x1 and x2 diffs and division by scalar
    # Matlab version has negative values here!
    # Combined at x1 and x2 diffs
    # l4 squared
    # add minus??!
    # with 2 lengthscales we need to divide by the product of the lengthscales squared
    upper_right = torch.prod(diff, dim = -1).div(lx1.square() * lx2.square())

    # same as other off-diagonal block
    lower_left = upper_right

    # Concatenate upper and lower blocks column-wise, and then concatenate them row-wise
    # torch.Size([2 * n_train, 2 * n_test])
    blocks = torch.cat((
        torch.cat((upper_left, upper_right), 1), 
        torch.cat((lower_left, lower_right), 1)
        ), 0).to(row_tensor.device)

    # torch.Size([2 * n_row, 2 * n_column])
    # elementwise multiplication
    # sum squared difference over x1 and x2, divide by -2 * l^2, and exponentiate. Tile for blocks
    # with 2 lengthscales again we need to move the division inde (defore summing) so we target the individual dims
    K = sigma_f.square() * blocks.mul(diff.square().div(l.square()).sum(dim = -1).div(-2).exp().tile(2, 2))

    return K.to(row_tensor.device)

def block_diagonal_se_kernel(
        row_tensor, # torch.Size([n_rows, 2])
        column_tensor, # torch.Size([n_columns, 2])
        hyperparameters):
    
    """
    Calculate the SE kernel for two sets of points in 2D space.
    The parameter B controls the cross correlation between the two outputs. For symmtry, the off-diagonal element B must be the same.
    R^2 -> R^2

    Inputs:
        row_tensor: torch.Size([n_rows, 2])
        column_tensor: torch.Size([n_columns, 2])
        hyperparameters: list of length 4 containing sigma_n, sigma_f and l, and B
        If B is fixed, sigma_f is needed. Otherwise this is overoptimised (redundant degree of freedom)

    Returns:
        K: torch.Size([n_rows * 2, n_columns * 2])
    """
    
    # block diagonal R^2 -> R^2
    # correlations between outputs are not considered, depending on B
    
    # Extract hyperparameters except for sigma_n
    # sigma_f_squared = torch.exp(hyperparameters[1]) # torch.exp(log_sigma_f_squared)
    # sigma_f_squared = hyperparameters[1]
    sigma_f = hyperparameters[1]
    l = hyperparameters[2]
    B = hyperparameters[3]

    # NOTE: this is technically not needed (l is squared), but we apply it for consistency
    l = torch.nn.functional.softplus(l, beta = 5.0) + 1e-8

    # ensuring that B is symmetric (i.e. the off diagnal elements must be the same for symmetry)
    B = (B + B.T) / 2

    # Accommodate for single or double lengthscale

    # add a dimension [n_rows, 1, 2]
    rows_expanded = row_tensor[:, None, :]
    # [1, n_columns, 2]
    columns_expanded = column_tensor[None, :, :] 

    # difference scaled by lengthscales (2D)
    # torch.Size([n_rows, n_columns, 2])
    # Mahalanobis Distance (scaled)
    # l can be torch.Size([1]) or torch.Size([2]): both works
    # l is squared here so we don't need to ensure positivity
    scaled_diff = (rows_expanded - columns_expanded) / l**2

    # square and reduce dimensions
    # torch.Size([n_rows, n_columns])
    sqdist = torch.sum(scaled_diff ** 2, dim = -1)

    # only one signal variance
    # torch.Size([n_rows, n_columns])
    K_SE = sigma_f.square() * torch.exp(-0.5 * sqdist)
    
    # B is 2 x 2 and defines the cross correlation
    # Kronecker produces block diagonal
    K = torch.kron(B, K_SE)

    return K

