import torch
import torch.nn as nn
from torch.func import jacfwd

##########################
### dfNN == NCL == HNN ###
##########################

class dfNN_for_vmap(nn.Module):
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