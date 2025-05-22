import torch
import torch.nn as nn

##########################
### dfNN == NCL == HNN ###
##########################

class dfNN_matrix(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim * input_dim  # flattened 2x2 matrix

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x):
        x.requires_grad_(True)

        M = self.net(x).view(-1, 2, 2)
        # Make antisymmetric matrix
        A = M - M.transpose(1, 2)

        u = A[:, 0, 1]  # since A = [[0, u], [-u, 0]]

        du_dx = torch.autograd.grad(u.sum(), x, create_graph = True)[0]  # [B, 2]

        symplectic = du_dx.flip(-1) * torch.tensor([1.0, -1.0], device = x.device)

        return symplectic 
    
class dfNN(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1  # Scalar potential

        # HACK: SiLu() worked much better than ReLU() for this model

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
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

        # return symp, H # NOTE: return H as well if we want to see what is going on
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

        # NOTE: Increasing hidden dim and number of layers did not help
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