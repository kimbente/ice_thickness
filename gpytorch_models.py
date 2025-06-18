import torch
import gpytorch

from configs import SIGMA_F_RANGE, SIGMA_F_FIXED_RESIDUAL_MODEL_RANGE, SIGMA_F_RESIDUAL_MODEL_RANGE, SIGMA_N_RANGE, L_RANGE

# setting device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##################
### df KERNEL ####
##################

class dfRBFKernel(gpytorch.kernels.Kernel):
    # NOTE: This is always for num_task = 2 i.e. 2D vector field
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # register trainable parameter and set initial (raw) lengthscale
        # the raw (unconstrained) parameter will later we transformed to enforce the constraint
        self.register_parameter(name = "raw_lengthscale",
                                parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0])))
        
         # register a constraint to ensure the lengthscale, self.lengthscale, is positive
        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())
        # NOTE: We wrap the output scalar around this base_kernel outside of this class
    
    # --- Properties for read-access (transforming raw parameters and call kernel.lengthscale) ---
    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
    
    # --- Setters to allow assigning transformed values rather than raw ---
    @lengthscale.setter
    def lengthscale(self, value):
        self.initialize(raw_lengthscale = self.raw_lengthscale_constraint.inverse_transform(value))

    def forward(self, row_tensor, column_tensor, diag = False, **params):
        """
        Args:
            row_tensor: torch.Size([N, 2]) first input will correspond to rows in returned K
            column_tensor: torch.Size([M, 2]) second input will correspond to columns in returned K
            diag: bool, if True, return only the diagonal of the covariance matrix. Needed for Quantile Coverage Error (QCE) calculation.
        Returns:
            K: torch.Size([2N, 2M]) block format covariance matrix
        """
        # Extract the chosen device
        device = row_tensor.device

        # Extract both lengthscales
        l1, l2 = self.lengthscale[0].to(device), self.lengthscale[1].to(device)

        # STEP 1: Pairwise differences of shape [N, M, 2]
        # Expand row_tensor [N, 2] -> [N, 1, 2] and column_tensor [M, 2] -> [1, M, 2]
        diff = (row_tensor[:, None, :] - column_tensor[None, :, :]).to(device)
        # Extract the components (columns) for convenience, matching paper notation
        r1 = diff[:, :, 0]
        r2 = diff[:, :, 1]

        # STEP 2: Block matrix
        # Compute the 4 (2x2) block components
        upper_left = l2.square() - r2.square()
        lower_right = l1.square() - r1.square()
        upper_right = r1 * r2
        lower_left = upper_right # symmetric

        # Assemble the 2x2 block matrix
        top = torch.cat((upper_left, upper_right), dim = 1) # Shape: [N, 2M]
        bottom = torch.cat((lower_left, lower_right), dim = 1) # Shape: [N, 2M]
        blocks = torch.cat((top, bottom), dim = 0) # Shape: [2N, 2M]

        # STEP 3: RBF/SE envelope (elementwise)
        exponent_term = torch.exp(-0.5 * ((r1 / l1) ** 2 + (r2 / l2) ** 2))  # Shape: [N, M]
        
        # .tile(2, 2) forms (N, M) -> (2N, 2M) for the 2D vector field
        K = (1 / (l1**2 * l2**2)) * blocks * exponent_term.tile(2, 2)

        # Add this for Quantile Coverage Error (QCE) calculation
        if diag:
            # Return only the diagonal as a 1D tensor
            return K.diag()

        # NOTE: This is the base kernel and not scaled yet
        return K

##################
### dfNN MEAN ####
##################

class dfNN(gpytorch.means.Mean):
    # NOTE: This needs to be initialised of class gpytorch.means.Mean
    def __init__(self, input_dim = 2, hidden_dim = 32):
        # NOTE: we use the same default dimensionalities as the dfNN NN model
        super(dfNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 1  # Scalar potential (corresponds to H in HNNs)
        
        # HACK: SiLu() worked much better than ReLU() for this gradient-based model

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, self.output_dim),
        )
    
    def forward(self, x):
        H = self.net(x)

        partials = torch.autograd.grad(
                outputs = H.sum(), # we can sum here because every H row only depend on every x row
                inputs = x,
                create_graph = True
            )[0]

        # Symplectic gradient
        # flip columns (last dim) for x2, x1 order. Multiply second column by -1
        mean_symp = partials.flip(-1) * torch.tensor([1, -1], dtype = torch.float32, device = x.device)

        # return symp, H 
        # # NOTE: return H as well if we want to see what is going on
        return mean_symp

#########################
### MULTITASK WRAPPER ###
#########################

from typing import Optional

from gpytorch.lazy import lazify
from gpytorch.kernels import Kernel

# HACK: Need custom wrapper to have it run
class MultitaskKernelWrapper(Kernel):
    r"""
    Custom multitask kernel wrapper for GPyTorch where the full sized kernel is defined by data_covar_module.

    :param ~gpytorch.kernels.Kernel data_covar_module: Kernel to use as the data kernel.
    :param int num_tasks: Number of tasks
    :param int rank: (default 1) Rank of index kernel to use for task covariance matrix.
    """

    def __init__(
        self,
        data_covar_module: Kernel,
        num_tasks: int,
        **kwargs,
    ):
        """"""
        super(MultitaskKernelWrapper, self).__init__(**kwargs)
        self.data_covar_module = data_covar_module
        self.num_tasks = num_tasks

    def forward(self, x1, x2, diag = False, last_dim_is_batch = False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        covar_x = lazify(self.data_covar_module.forward(x1, x2, **params))
        return covar_x.diag() if diag else covar_x

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks

##############  
### MODELS ###
##############

############
### dfGP ###
############

from configs import L_RANGE, SIGMA_F_RANGE, SIGMA_N_RANGE

class dfGP(gpytorch.models.ExactGP):
    # dfGP model with zero mean
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks = 2
            )

        self.base_kernel = MultitaskKernelWrapper(
            dfRBFKernel(), 
            num_tasks = 2,
            )

        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        
        # initialize hyperparameters by sampling from a uniform distribution over predefined ranges
        self.base_kernel.data_covar_module.lengthscale = torch.empty(2, device = device).uniform_( * L_RANGE)
        self.covar_module.outputscale = torch.empty(1, device = device).uniform_( * SIGMA_F_RANGE)
        self.likelihood.task_noises = torch.empty(2, device = device).uniform_( * SIGMA_N_RANGE)

        # add constraint to likelihood 1e-4 is the default
        # self.likelihood.raw_task_noises_constraint = gpytorch.constraints.GreaterThan(1e-5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # NOTE: Assure it is multitask
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

##############
### dfGPcm ###
##############

class dfGPcm(gpytorch.models.ExactGP):
    # dfGP model with zero mean
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks = 2
            )
        # NOTE: Initialise/view with
        # model.mean_module.base_means[0].initialize(constant = 1.0) OR
        # model.mean_module.base_means[0].constant.item()

        self.base_kernel = MultitaskKernelWrapper(
            dfRBFKernel(), 
            num_tasks = 2,
            )

        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        
        # initialize hyperparameters by sampling from a uniform distribution over predefined ranges
        self.base_kernel.data_covar_module.lengthscale = torch.empty(2, device = device).uniform_( * L_RANGE)
        self.covar_module.outputscale = torch.empty(1, device = device).uniform_( * SIGMA_F_RANGE)
        self.likelihood.task_noises = torch.empty(2, device = device).uniform_( * SIGMA_N_RANGE)

        # add constraint to likelihood 1e-4 is the default
        # self.likelihood.raw_task_noises_constraint = gpytorch.constraints.GreaterThan(1e-5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # NOTE: Assure it is multitask
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

#############   
### dfNGP ###
#############

class dfNGP(gpytorch.models.ExactGP):
    # dfGP model with constant mean
    def __init__(self, train_x, train_y, likelihood):
        super(dfNGP, self).__init__(train_x, train_y, likelihood)

        # Custom mean module
        self.mean_module = dfNN(input_dim = 2) # default hidden_dim = 32
        # Custom kernel module
        self.base_kernel = MultitaskKernelWrapper(
            dfRBFKernel(), 
            num_tasks = 2,
            )
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

        # initialise
        self.covar_module.outputscale = torch.empty(1, device = device).uniform_( * SIGMA_F_RANGE)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

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