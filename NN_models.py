import torch
import torch.nn as nn
from torch.func import jacrev, jacfwd

################
### NCL/dfNN ###
################

class dfNN_for_vmap(nn.Module):
    # for a single point
    def __init__(self, input_dim = 2, hidden_dim = 32):
        super().__init__()

        self.input_dim = input_dim
        
        # Output dim follows input dims
        # for 2D input the NN output dim is 1
        self.output_dim = int((input_dim * (input_dim - 1)) / 2)

        # Replace with something more sophisticated
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, x):
        # put deterministic transformations here with torch functional

        def A(x):

            # RUN THROUGH NET
            U_fill = self.net(x)
        
            # This version works with vmap, diagonal shifts diagional one up
            U_zero_one = torch.triu(torch.ones(self.input_dim, self.input_dim), diagonal = 1)

            # put on same device as U_fill
            U_zero_one = U_zero_one.to(U_fill.device) 

            # U_zero_one is (2, 2), U_fill is (1), so we can just scale U_zero_one by U_fill
            U = U_zero_one * U_fill

            # U is (2, 2), so we need to swap the last two dims and then subtract
            A = U - U.T

            return A

        return torch.diagonal(jacfwd(A)(x), dim1 = 1, dim2 = 2).sum(-1)

############################
### NCL/dfNN full matrix ###
############################

class dfNN_fullmatrix_for_vmap(nn.Module):
    # for a single point
    def __init__(self, input_dim = 2, hidden_dim = 32):
        super().__init__()

        self.input_dim = input_dim
        
        # Output dim follows input dims
        # for 2D input the NN output dim is now 4 (2x2) in the full matrix case
        self.output_dim = int((input_dim * input_dim))

        # Replace with something more sophisticated
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, x):
        # put deterministic transformations here with torch functional

        def A(x):

            # RUN THROUGH NET
            M = self.net(x)

            # make square
            M = M.view(self.input_dim, self.input_dim)

            # construct an Anti-symmetric matrix from the output of the NN
            # the diagonal of the matrix M is irrelevant
            A = M - M.mT

            return A

        return torch.diagonal(jacfwd(A)(x), dim1 = 1, dim2 = 2).sum(-1)

##################################
### Hamiltonian Neural Network ###
##################################

class HNN_for_vmap(nn.Module):
    # for a single point
    def __init__(self, input_dim = 2, hidden_dim = 32):
        super().__init__()

        self.input_dim = input_dim
        
        # Scalar output: maps R^2 -> R
        self.output_dim = int(1)

        # Replace with something more sophisticated
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, x):
        # put deterministic transformations here with torch functional

        def H(x):

            # RUN THROUGH NET
            # Retrieve scalar potential
            H = self.net(x)

            return H

        # The Jacobian of H has shape torch.Size([1, 2]) (1 batch, 2 output dims)
        # reshape to torch.Size([2])
        jac_H = jacfwd(H)(x).squeeze()

        # flip the order of the output dims and multiply second dim by -1
        return jac_H.flip(0) * torch.tensor([1, -1], device = jac_H.device)

####################
### MLP for PINN ###
####################

class PINN_backbone(nn.Module):
    # for a single point
    def __init__(self, input_dim = 2, hidden_dim = 32):
        super().__init__()

        self.input_dim = input_dim
        
        # Output dim follows input dims
        # for 2D input the NN output dim is 2D
        self.output_dim = input_dim

        # Replace with something more sophisticated
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, x):
        # return 2D field
        return self.net(x)