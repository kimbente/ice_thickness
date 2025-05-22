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

### HELPER FUNCTIONS FOR MORE DETAILED SIMULATIONS

def get_pdf_of_gaussian_component_long(x, mu, sigma):
    """ Caluclate the probability density function (pdf) of a 2D Gaussian component at each point.
    p(x) = (1 / (2 * π * sqrt(det(Σ)))) * exp(-0.5 * (x - μ)ᵀ Σ⁻¹ (x - μ))

    Args:
        x (torch.Size([n, 2])): x points where the last two dims are the x and y coordinates
        mu (torch.Size([2])): mean location of the Gaussian component, should be within (or near) x domain of course
        sigma (torch.Size([2, 2])): Covariance matrix of the Gaussian component

    Returns:
        pdf (torch.Size([n])): pdf of the Gaussian component at each point 
    """
    diff = x - mu # shape: (n, 2)

    inv_sigma = torch.linalg.inv(sigma) # shape: (2, 2) - invert so we can multiply
    det_sigma = torch.linalg.det(sigma) # shape: (1,) - determinant is a scalar

    # Mahalanobis distance: (x - μ)ᵀ Σ⁻¹ (x - μ), for each x in the grid
    # Step 1: Matmul to get the intermediate result (N, D) @ (D, D) → (N, D)
    # Step 2: Mul is element-wise multiplication
    # Step 3: Sum over the last dimension to get the final result (N, D) → (N, 1)
    exponent = -0.5 * torch.sum(torch.mul(torch.matmul(diff, inv_sigma), diff), dim = 1)

    # sqrt(det(Σ)) is the square root of the determinant of the covariance matrix Σ.
    norm_multiplier = 1 / (2 * torch.pi * torch.sqrt(det_sigma))

    # Put it all together
    pdf = norm_multiplier * torch.exp(exponent) # shape: (n, 1)

    return pdf

def compose_unnormalised_gaussian_mixture_long(x, mus, sigmas, weights):
    """ Compose an unnormalised Gaussian mixture (UGM) from multiple Gaussian components. Returns a vector of the same length as x.

    Args:
        x (torch.Size([n, 2])): list of n points in 2D where the last two dims are the x and y coordinates
        mus (list of torch.Size([2])): list of mean locations of the Gaussian components
        sigmas (list of torch.Size([2, 2])): list of covariance matrices of the Gaussian components
        weights (list of float): list of weights for each Gaussian component

    Returns:
        pdf (torch.Size([n])): pdf of the UGM at each point
    """
    # Extract length for intialisation of placeholder
    n = x.shape[0] 

    # Unnormalised Gaussian Mixture Model (UGM) is a weighted sum of the individual Gaussian components
    ugm = torch.zeros(n) # shape: (n, )

    for mu, sigma, weight in zip(mus, sigmas, weights):
        ugm += weight * get_pdf_of_gaussian_component_long(x, mu, sigma)

    return ugm

def get_directed_bls_long(x, angle_degree):
    """
    Generate a linear stream function (scalar field) based on the x coordinates
    The stream function is directed in a specified angle (in degrees).
    Args:
        x (torch.Size([n, 2])): x coordinate list where the last two dims are the x and y coordinates
        angle_degree (float): angle in degrees for the direction of the stream function
    Returns:
        directed_stream (torch.Size([n])): directed stream function at each point
    """
    # Convert the angle from degrees to radians
    angle_rad = torch.deg2rad(torch.tensor(angle_degree))

    a = torch.cos(angle_rad)  # x weight
    b = torch.sin(angle_rad)  # y weight

    # stream(x, y) = a * x + b * y
    directed_stream = a * x[:, 0] + b * x[:, 1] # shape: (n)
    
    return directed_stream

def combine_bls_and_ugm(bls, ugm, ugm_weight = 1.0):
    """ Combine the BLS and UGM grids to get the final grid. Both grids are assumed to be the same size.

    Args:
        bls (torch.Size([n_side, n_side]) or torch.Size([n])): BLS grid/long list
        ugm (torch.Size([n_side, n_side]) or torch.Size([n]): UGM grid/long list

    Returns:
        combined_grid (torch.Size([n_side, n_side]) or torch.Size([n]): combined grid/long list
    """
    # Combine the BLS and UGM grids/components
    combined = bls + (ugm_weight * ugm)

    return combined

def get_vector_field_from_stream_long(x, psi):
    """ We use autograd to calculate the partial derivatives of the stream function and then compose the curl, i.e. the divergence-free vector field from it. 

    Note: 
    - Computing the full Jacobian is not necessary ("Jacobian overkill"), since we only need the partial derivatives of the stream function with respect to x and y. 
    - While at first glance it appears suprising that we can use the psi.sum() function to get a scalar output, this is because the gradient of a scalar function with respect to a vector is a vector, and psi[i, j] only depends on x[i, j], so the gradient is non-zero only for the i, j-th element of the x tensor.

    This would yield the same result as:
    grad_psi = torch.autograd.grad(
        outputs = psi, # non-scalar output
        inputs = x_grid,
        grad_outputs = torch.ones_like(psi)) 

    More background is provided here: https://discuss.pytorch.org/t/need-help-computing-gradient-of-the-output-with-respect-to-the-input/150950/4

    Alternative: torch.gradient() is based on finite differences, so it is not a perfect gradient operator.

    Args:
        x (torch.Size([n, 2])): x coordinate list where the last two dims are the x and y coordinates. Must have had requires_grad = True!!
        psi (torch.Size([n])): psi stream function generated from x

    Returns:
        vector_field (torch.Size([n, 2])): vector field at each point
    """
    grad_psi = torch.autograd.grad(
        outputs = psi.sum(), # scalar, but only depends on corresponding x[i, j]
        inputs = x, 
        )[0] # index 0 because we only have one input tensor

    # Calculate the vector field (curl) from the gradients: dψ/dy and -dψ/dx
    vector_field = torch.stack([grad_psi[:, 1], - grad_psi[:, 0]], dim = -1) # shape: (n, 2)

    return vector_field

#################
### FUNTION 1 ###
#################
# rather smooth example

def simulate_detailed_convergence(x_inputs):
    # mean magnitude is 1.3

    ### Fixed ###
    list_of_mus = [
        torch.tensor([0.2, 0.3]),
        torch.tensor([0.5, 1.0]),
        ]

    list_of_sigmas = [
        torch.tensor([[0.05, 0.0], [0.01, 0.05]]),
        torch.tensor([[0.2, 0.0], [0.0, 0.1]]),
        ]

    list_of_weights = [
        -0.2, 
        1.2,
        ] 

    ugm_weight = 0.5
    
    ###
    x_inputs = x_inputs.requires_grad_() # Set requires_grad to True for gradients i.e. to calculate the curl
    
    # produce Unnormalised Gaussian Mixture component 
    ugm = compose_unnormalised_gaussian_mixture_long(
        x_inputs,
        list_of_mus,
        list_of_sigmas,
        list_of_weights
    ) # shape: (n)
    
    # produce baseline stream field
    bls = get_directed_bls_long(x_inputs, angle_degree = 90) # shape: (n)

    # stream function: weighted sum of the BLS and UGM
    psi = combine_bls_and_ugm(bls, ugm, ugm_weight = ugm_weight) # shape: (n)

    psi_vector_field = get_vector_field_from_stream_long(x_inputs, psi) # shape: (n, 2)

    simulated_vector_field = simulate_convergence(x_inputs.detach().clone()) # torch.Size([n, 2])

    vector_field = psi_vector_field * 0.4 + simulated_vector_field # torch.Size([n, 2])

    return vector_field.detach().clone()

#################
### FUNTION 2 ###
#################

def simulate_detailed_deflection(x_inputs):
    # mean magnitude is 1.47
    ### Fixed ###
    list_of_mus = [
        torch.tensor([0.3, 0.7]),
        torch.tensor([0.9, 0.1]),
        torch.tensor([0.05, 0.05]),
        ]

    # Covariance matrices for each Gaussian component
    list_of_sigmas = [
        torch.tensor([[0.05, 0.02], [0.002, 0.01]]),
        torch.tensor([[0.2, 0.0], [0.0, 0.1]]),
        torch.tensor([[0.05, 0.0], [0.0, 0.05]]),
        ]

    list_of_weights = [
        0.2, 
        -1.5,
        1.0
        ] 

    ugm_weight = 0.2
    
    ###
    x_inputs = x_inputs.requires_grad_() # Set requires_grad to True for gradients i.e. to calculate the curl
    
    # produce Unnormalised Gaussian Mixture component 
    ugm = compose_unnormalised_gaussian_mixture_long(
        x_inputs,
        list_of_mus,
        list_of_sigmas,
        list_of_weights
    ) # shape: (n)
    
    # produce baseline stream field
    bls = get_directed_bls_long(x_inputs, angle_degree = 90) # shape: (n)

    # stream function: weighted sum of the BLS and UGM
    psi = combine_bls_and_ugm(bls, ugm, ugm_weight = ugm_weight) # shape: (n)

    psi_vector_field = get_vector_field_from_stream_long(x_inputs, psi) # shape: (n, 2)

    simulated_vector_field = simulate_deflection(x_inputs.detach().clone()) # torch.Size([n, 2])

    vector_field = psi_vector_field * 0.4 + simulated_vector_field # torch.Size([n, 2])

    return vector_field.detach().clone()

#################
### FUNTION 3 ###
#################

def simulate_detailed_curve(x_inputs, return_components_bool = False):
    # mean magnitude is 1.57

    ### Fixed ###
    list_of_mus = [
        torch.tensor([0.4, 0.7]),
        torch.tensor([0.2, -0.05]),
        ]

    # Covariance matrices for each Gaussian component
    # Smaller: Lower variance (more concentrated) in that direction
    # Off-diagonal elements: Correlation between x and y
    list_of_sigmas = [
        torch.tensor([[0.05, 0.03], [0.002, 0.01]]),
        torch.tensor([[0.05, 0.0], [0.0, 0.05]]),
        ]

    list_of_weights = [
        0.2, 
        -1.0
        ] 

    ugm_weight = 0.5
    
    ###
    x_inputs = x_inputs.requires_grad_() # Set requires_grad to True for gradients i.e. to calculate the curl
    
    # produce Unnormalised Gaussian Mixture component 
    ugm = compose_unnormalised_gaussian_mixture_long(
        x_inputs,
        list_of_mus,
        list_of_sigmas,
        list_of_weights
    ) # shape: (n)
    
    # produce baseline stream field
    bls = get_directed_bls_long(x_inputs, angle_degree = 90) # shape: (n)

    # stream function: weighted sum of the BLS and UGM
    psi = combine_bls_and_ugm(bls, ugm, ugm_weight = ugm_weight) # shape: (n)

    psi_vector_field = get_vector_field_from_stream_long(x_inputs, psi) # shape: (n, 2)

    simulated_vector_field = simulate_merge(x_inputs.detach().clone()) # torch.Size([n, 2])

    vector_field = psi_vector_field * 0.2 + simulated_vector_field # torch.Size([n, 2])

    if return_components_bool:
        return vector_field.detach().clone(), psi, psi_vector_field, simulated_vector_field
    else:
        return vector_field.detach().clone()

#################
### FUNTION 4 ###
#################

def simulate_detailed_ridges(x_inputs):
    # mean magnitude is 1.47

    ### Fixed ###
    list_of_mus = [
        torch.tensor([0.2, 0.3]),
        torch.tensor([0.5, 1.0]),
        torch.tensor([1.0, 0.7]),
        torch.tensor([1.0, 0.2]),
        ]

    list_of_sigmas = [
        torch.tensor([[0.05, 0.0], [0.01, 0.1]]),
        torch.tensor([[0.2, 0.0], [0.0, 0.1]]),
        torch.tensor([[0.3, 0.0], [0.0, 0.01]]),
        torch.tensor([[0.3, 0.0], [0.0, 0.01]]),
        ]

    list_of_weights = [
        -0.2, 
        1.2,
        0.2,
        0.2
        ] 

    ugm_weight = 0.1
    
    ###
    x_inputs = x_inputs.requires_grad_() # Set requires_grad to True for gradients i.e. to calculate the curl
    
    # produce Unnormalised Gaussian Mixture component 
    ugm = compose_unnormalised_gaussian_mixture_long(
        x_inputs,
        list_of_mus,
        list_of_sigmas,
        list_of_weights
    ) # shape: (n)

    # different way of producing baseline streamfield
    bls = (x_inputs[:, 0] + 0.2 * x_inputs[:, 1])**3 # shape: (n)
                          
    # stream function: weighted sum of the BLS and UGM
    psi = combine_bls_and_ugm(bls, ugm, ugm_weight = ugm_weight) # shape: (n)

    psi_vector_field = get_vector_field_from_stream_long(x_inputs, psi) # shape: (n, 2)

    # no simulated vector field needed

    return psi_vector_field.detach().clone()

##################
### FUNCTION 5 ###
##################

def simulate_detailed_branching(x_inputs):
    # mean magnitude is 1.0

    list_of_mus = [
        torch.tensor([0.0, -0.1]),
        torch.tensor([0.5, 1.0]),
        torch.tensor([1.0, 0.7]),
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.8, 0.2]),
        ]

    list_of_sigmas = [
        torch.tensor([[0.003, 0.0], [0.0, 0.04]]),
        torch.tensor([[0.05, 0.0], [0.0, 0.05]]),
        torch.tensor([[0.3, 0.0], [0.0, 0.01]]),
        torch.tensor([[0.006, 0.0], [0.008, 0.004]]),
        torch.tensor([[0.004, 0.004], [0.0, 0.006]]),
        ]

    list_of_weights = [
        - 0.8, 
        0.5,
        0.3,
        0.03,
        -0.03
        ] 

    ugm_weight = 0.2
    
    ###
    x_inputs = x_inputs.requires_grad_() # Set requires_grad to True for gradients i.e. to calculate the curl
    
    # produce Unnormalised Gaussian Mixture component 
    ugm = compose_unnormalised_gaussian_mixture_long(
        x_inputs,
        list_of_mus,
        list_of_sigmas,
        list_of_weights
    ) # shape: (n)

    # produce baseline stream field
    bls = get_directed_bls_long(x_inputs, angle_degree = 1) # shape: (n)

    # stream function: weighted sum of the BLS and UGM
    psi = combine_bls_and_ugm(bls, ugm, ugm_weight = ugm_weight) # shape: (n)

    psi_vector_field = get_vector_field_from_stream_long(x_inputs, psi) # shape: (n, 2)

    simulated_vector_field = simulate_branching(x_inputs.detach().clone()) # torch.Size([n, 2])

    vector_field = psi_vector_field * 0.1 + simulated_vector_field # torch.Size([n, 2])

    return vector_field.detach().clone()

def simulate_detailed_edge(x_inputs):
    # Purely constructed from psi
    # sigmoid for stronger edge

    list_of_mus = [
        torch.tensor([0.15, 0.5]), # first diagonal riffle
        torch.tensor([0.7, 0.9]), # top right deterrent
        torch.tensor([0.5, 0.5]), # big central diagonal
        torch.tensor([0.8, 0.4]), # bottom right negative
        ]

    list_of_sigmas = [
        torch.tensor([[0.003, 0.0], [0.005, 0.003]]), 
        torch.tensor([[0.02, 0.001], [0.0, 0.01]]),
        torch.tensor([[0.006, 0.0], [0.008, 0.004]]), # diag
        torch.tensor([[0.006, 0.004], [0.002, 0.006]]),
        ]

    list_of_weights = [
        0.005, 
        0.1,
        0.02,
        -0.01
        ] 
    
    ugm_weight = 0.1
    
    x_inputs = x_inputs.requires_grad_() # Set requires_grad to True for gradients i.e. to calculate the curl
    
    ugm = compose_unnormalised_gaussian_mixture_long(
        x_inputs,
        list_of_mus,
        list_of_sigmas,
        list_of_weights
    )

    # baseline stream with sigmoid to make it more pronounced
    bls = torch.sigmoid(((get_directed_bls_long(x_inputs, angle_degree = 100)) * 6) - 4)
    
    # stream function
    psi = combine_bls_and_ugm(bls, ugm, ugm_weight)

    vector_field = get_vector_field_from_stream_long(x_inputs, psi)

    return vector_field.detach().clone()

    