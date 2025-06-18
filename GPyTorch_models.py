import torch
import gpytorch
from gpytorch.kernels import Kernel

from configs import SIGMA_F_RANGE, SIGMA_F_FIXED_RESIDUAL_MODEL_RANGE, SIGMA_F_RESIDUAL_MODEL_RANGE, SIGMA_N_RANGE, L_RANGE

# setting device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############
### KERNEL ####
###############
class DivergenceFreeSEKernel(Kernel):
    """
    Divergence-free squared exponential (SE) kernel in 2D.

    Returns a (2N x 2M) matrix for 2D vector fields, ensuring divergence-free structure.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Register the outputscale / variance (sigma_f) as a learnable parameter
        self.register_parameter(name = "raw_outputscale", 
                                parameter = torch.nn.Parameter(torch.tensor(1.0)))
        
        self.register_parameter(name = "raw_lengthscale",
                                parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0])))

        # Register transform for positivity (softplus)
        self.register_constraint("raw_outputscale", gpytorch.constraints.Positive())

        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)
    
    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @outputscale.setter
    def outputscale(self, value):
        self.initialize(raw_outputscale = self.raw_outputscale_constraint.inverse_transform(value))

    @lengthscale.setter
    def lengthscale(self, value):
        self.initialize(raw_lengthscale = self.raw_lengthscale_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag = False, **params):
        """
        Args:
            x1: torch.Size([2N, 1]) flattened, second explicit dim is automatic
            x2: torch.Size([2M, 1])
        Returns:
            K: torch.Size([2N, 2M])
        """
        # Transform long/flat format into 2D
        mid_x1 = x1.shape[0] // 2
        mid_x2 = x2.shape[0] // 2 

        # torch.Size([N, 2])
        x1 = torch.cat((x1[:mid_x1], x1[mid_x1:]), dim = 1).to(x1.device)
        # torch.Size([M, 2])
        x2 = torch.cat((x2[:mid_x2], x2[mid_x2:]), dim = 1).to(x2.device)

        l = self.lengthscale.squeeze().to(x1.device)  # Shape (2,)

        lx1, lx2 = l[0].to(x1.device), l[1].to(x1.device)

        sigma_f = self.outputscale

        # Broadcast pairwise differences: shape [N, M, 2]
        diff = (x1[:, None, :] - x2[None, :, :]).to(x1.device)

        ### 2x2 block components ###
        upper_left = (1 - diff[:, :, 1].square() / lx2.square()) / lx2.square()
        lower_right = (1 - diff[:, :, 0].square() / lx1.square()) / lx1.square()
        upper_right = (diff[:, :, 0] * diff[:, :, 1]) / (lx1.square() * lx2.square())
        lower_left = upper_right

        # Block matrix assembly
        top = torch.cat((upper_left, upper_right), dim = 1)
        bottom = torch.cat((lower_left, lower_right), dim = 1)
        blocks = torch.cat((top, bottom), dim = 0)

        # RBF/SE envelope (elementwise)
        exp_term = torch.exp(-0.5 * (diff.square() / l.square()).sum(dim = -1))
        # .tile(2, 2) forms (N, M) -> (2N, 2M) for the 2D vector field
        K = sigma_f.square() * blocks * exp_term.tile(2, 2)

        # Add this for Quantile Coverage Error (QCE) calculation
        if diag:
        # Return only the diagonal as a 1D tensor
            return K.diag()

        return K

##############  
### MODELS ###
##############

############
### dfGP ###
############

class dfGP(gpytorch.models.ExactGP):
    # dfGP model with zero mean
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = DivergenceFreeSEKernel()
        
        # initialize hyperparameters by sampling from a uniform distribution over predefined ranges
        self.likelihood.noise = torch.empty(1, device = device).uniform_( * SIGMA_N_RANGE)
        self.covar_module.outputscale = torch.empty(1, device = device).uniform_( * SIGMA_F_RANGE)
        self.covar_module.lengthscale = torch.empty(2, device = device).uniform_( * L_RANGE)

        self.likelihood.noise_covar.register_constraint(
            "raw_noise", gpytorch.constraints.GreaterThan(1e-4)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

##############
### dfGPcm ###
##############

class TwoBlockConstantMean(gpytorch.means.ConstantMean):
    # Default is a 2D tensor with zeros for each block
    def __init__(self, constant_prior = torch.tensor([0.0, 0.0])):
        # Initialize parent ConstantMean for the first constant
        super().__init__()

        # learnable constant mean, initialised as the mean of all training data
        self.register_parameter("constant_x", torch.nn.Parameter(torch.tensor([constant_prior[0]])))
        self.register_parameter("constant_y", torch.nn.Parameter(torch.tensor([constant_prior[1]])))


    def forward(self, input):
        """
        input: shape (2N)
        Returns: shape (2N)
        """
        N = input.shape[0] // 2

        # Expand constant_x and constant_y for their blocks
        mean_x = self.constant_x.expand(N, 1)
        mean_y = self.constant_y.expand(N, 1)
        # Concatenate to form (N, 2) mean
        mean = torch.cat((mean_x, mean_y), dim = 1).squeeze()
        # HACK: Reshape to interleaved format. This is counterintuitive but necessary
        mean = mean.reshape(-1)

        return mean

class dfGPcm(gpytorch.models.ExactGP):
    # dfGP model with constant mean
    def __init__(self, train_x, train_y, likelihood, constant_mean):
        # Inherit from ExactGP with 3 inputs + self = 4 inputs
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = TwoBlockConstantMean(constant_mean)
        self.covar_module = DivergenceFreeSEKernel()
        
        # initialize hyperparameters by sampling from a uniform distribution over predefined ranges
        self.likelihood.noise = torch.empty(1, device = device).uniform_( * SIGMA_N_RANGE)
        self.covar_module.outputscale = torch.empty(1, device = device).uniform_( * SIGMA_F_FIXED_RESIDUAL_MODEL_RANGE)
        self.covar_module.lengthscale = torch.empty(2, device = device).uniform_( * L_RANGE)

        self.likelihood.noise_covar.register_constraint(
            "raw_noise", gpytorch.constraints.GreaterThan(1e-4)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#############   
### dfNGP ###
#############

from NN_models import dfNN

class dfNGP(gpytorch.models.ExactGP):
    # dfGP model with constant mean
    def __init__(self, train_x, train_y, likelihood):
        # Inherit from ExactGP with 3 inputs + self = 4 inputs
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = dfNN()
        self.covar_module = DivergenceFreeSEKernel()
        
        # initialize hyperparameters by sampling from a uniform distribution over predefined ranges
        self.likelihood.noise = torch.empty(1, device = device).uniform_( * SIGMA_N_RANGE)
        self.covar_module.outputscale = torch.empty(1, device = device).uniform_( * (SIGMA_F_RESIDUAL_MODEL_RANGE))
        self.covar_module.lengthscale = torch.empty(2, device = device).uniform_( * L_RANGE)

        self.likelihood.noise_covar.register_constraint(
            "raw_noise", gpytorch.constraints.GreaterThan(1e-4)
        )

    def forward(self, x):
        # Make 2D for dfNN
        x_for_dfNN = x.reshape(2, -1).T
        mean_x_from_dfNN = self.mean_module(x_for_dfNN)
        # HACK: Reshape to interleaved format. This is counterintuitive but necessary
        mean_x = mean_x_from_dfNN.reshape(-1)

        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

##########   
### GP ###
##########

import gpytorch
from gpytorch.kernels import Kernel
import torch

class BlockStructureSEKernel(Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # lengthscale: vector of 2
        self.register_parameter("raw_lengthscale", torch.nn.Parameter(torch.tensor([1.0, 1.0])))
        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())

        # B diagonal entries
        self.register_parameter("raw_B_diagonal", torch.nn.Parameter(torch.tensor([1.0, 1.0])))
        self.register_constraint("raw_B_diagonal", gpytorch.constraints.Positive())

        # B off-diagonal (no constraint by default, can be negative!)
        self.register_parameter("raw_B_offdiagonal", torch.nn.Parameter(torch.tensor(1.0)))

    # --- Properties for read-access (transforming raw parameters) ---
    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @property
    def B_diagonal(self):
        return self.raw_B_diagonal_constraint.transform(self.raw_B_diagonal)

    @property
    def B_offdiagonal(self):
        return self.raw_B_offdiagonal  # no constraint

    # --- Setters to allow assigning transformed values ---
    @lengthscale.setter
    def lengthscale(self, value):
        self.initialize(raw_lengthscale = self.raw_lengthscale_constraint.inverse_transform(value))

    @B_diagonal.setter
    def B_diagonal(self, value):
        self.initialize(raw_B_diagonal = self.raw_B_diagonal_constraint.inverse_transform(value))

    @B_offdiagonal.setter
    def B_offdiagonal(self, value):
        self.initialize(raw_B_offdiagonal = value)  # no transform needed

    def forward(self, x1, x2, diag = False, **params):
        """
        Args:
            x1: torch.Size([2N, 1]) flattened, second explicit dim is automatic
            x2: torch.Size([2M, 1])
        Returns:
            K: torch.Size([2N, 2M])
        """
        # Transform long/flat format into 2D
        mid_x1 = x1.shape[0] // 2
        mid_x2 = x2.shape[0] // 2 

        # torch.Size([N, 2])
        x1 = torch.cat((x1[:mid_x1], x1[mid_x1:]), dim = 1).to(x1.device)
        # torch.Size([M, 2])
        x2 = torch.cat((x2[:mid_x2], x2[mid_x2:]), dim = 1).to(x2.device)

        l = self.lengthscale.squeeze().to(x1.device)  # Shape (2,)

        # Broadcast pairwise differences: shape [N, M, 2]
        diff = (x1[:, None, :] - x2[None, :, :]).to(x1.device)

        # RBF/SE envelope (elementwise)
        # NOTE: K_SE is unscaled to avoid redundant optimisation
        K_SE = torch.exp(-0.5 * (diff.square() / l.square()).sum(dim = -1))

        # Apply softplus or other positive constraint to diagonals
        b_diag = self.raw_B_diagonal_constraint.transform(self.raw_B_diagonal)
        # cross correlation may be negative
        b_offdiag = self.raw_B_offdiagonal

        # construct 2 x 2 B
        B = torch.tensor([
            [b_diag[0], b_offdiag],
            [b_offdiag, b_diag[1]]], device = x1.device, dtype = x1.dtype)
        
        # NOTE: Should technically be symmetric by definition but was numerically slightly off
        # ensuring that B is symmetric (i.e. the off diagnal elements must be the same for symmetry)
        B = (B + B.T) / 2

        # B is 2 x 2 and defines the correlation & cross correlation between u and v
        # Kronecker produces block diagonal
        K = torch.kron(B, K_SE)

        # Add this for Quantile Coverage Error (QCE) calculation
        if diag:
        # Return only the diagonal as a 1D tensor
            return K.diag()

        return K
    
from configs import SIGMA_N_RANGE, L_RANGE, B_DIAGONAL_RANGE, B_OFFDIAGONAL_RANGE

class GP(gpytorch.models.ExactGP):
    # dfGP model with constant mean
    def __init__(self, train_x, train_y, likelihood):
        # Inherit from ExactGP with 3 inputs + self = 4 inputs
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = BlockStructureSEKernel()
        
        # initialize hyperparameters by sampling from a uniform distribution over predefined ranges
        self.likelihood.noise = torch.empty(1, device = device).uniform_( * SIGMA_N_RANGE)
        self.covar_module.lengthscale = torch.empty(2, device = device).uniform_( * L_RANGE)
        self.covar_module.B_diagonal = torch.empty(2, device = device).uniform_( * B_DIAGONAL_RANGE)
        self.covar_module.B_offdiagonal = torch.empty(1, device = device).uniform_( * B_OFFDIAGONAL_RANGE)

        self.likelihood.noise_covar.register_constraint(
            "raw_noise", gpytorch.constraints.GreaterThan(1e-4)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)