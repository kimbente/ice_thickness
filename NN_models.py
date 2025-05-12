import torch
import torch.nn as nn

##########################
### dfNN == NCL == HNN ###
##########################
    
class dfNN(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1  # Scalar potential

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x):
        """
        Turn x1, x2 locations into vector fields
        x: [batch_size, input_dim]
        Returns: [batch_size, input_dim]  # Symplectic gradient
        """
        # Retrieve scalar potential
        H = self.net(x)

        partials = torch.autograd.grad(
                outputs = H.sum(), # we can sum here because every H row only depend on every x row
                inputs = x,
                create_graph = True
            )[0]
        
        # Symplectic gradient
        # flip columns (last dim) for x2, x1 order. Multiply x2 by -1
        symp = partials.flip(-1) * torch.tensor([1, -1], dtype = torch.float32, device = x.device)

        return symp

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