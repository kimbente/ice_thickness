import torch
import gpytorch

# For the kernel
from linear_operator.operators import to_linear_operator 

from configs import L_RANGE, SIGMA_F_RANGE, SIGMA_F_RESIDUAL_MODEL_RANGE, SIGMA_N_RANGE

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
        Computes the covariance matrix for a 2D vector field using the dfRBF kernel. Returns the covariance matrix in an interleaved format suitable for multitask Gaussian processes in GPyTorch.

        Args:
            row_tensor: torch.Size([N, 2]) first input will correspond to rows in returned K
            column_tensor: torch.Size([M, 2]) second input will correspond to columns in returned K
            diag: bool, if True, return only the diagonal of the covariance matrix. Needed for Quantile Coverage Error (QCE) calculation. Defaults to False.

        Returns:
            K: torch.Size([2N, 2M]) interleaved format covariance matrix
        """
        # Extract the chosen device
        device = row_tensor.device

        # Extract both lengthscales
        l1, l2 = self.lengthscale[0].to(device), self.lengthscale[1].to(device)

        # STEP 1: Pairwise differences of shape [N, M, 2]
        # Expand row_tensor [N, 2] -> [N, 1, 2] and column_tensor [M, 2] -> [1, M, 2]
        diff = (row_tensor[:, None, :] - column_tensor[None, :, :]).to(device)

        # Extract the relative components (columns of diff) for convenience, matching paper notation
        r1 = diff[:, :, 0]
        r2 = diff[:, :, 1]
        
        # STEP 2: Block matrix
        # Block components (shape N × M each)
        K_uu = (1 - (r2**2 / l2**2)) / l2**2
        K_uv = (r1 * r2) / (l1**2 * l2**2)
        K_vu = K_uv  
        K_vv = (1 - (r1**2 / l1**2)) / l1**2

        # STEP 3: RBF/SE envelope (elementwise) (shape N × M)
        exp_term = torch.exp(-0.5 * ((r1 / l1) ** 2 + (r2 / l2) ** 2))

        # STEP 4: Combine and stack
        # Final scaled components (each shape N × M)
        K_uu = K_uu * exp_term
        K_uv = K_uv * exp_term
        K_vu = K_vu * exp_term
        K_vv = K_vv * exp_term

        # Now interleave rows and columns
        # Stack into shape (N, M, 2, 2)
        K_blocks = torch.stack([
            torch.stack([K_uu, K_uv], dim = -1),
            torch.stack([K_vu, K_vv], dim = -1)
        ], dim = -2)  # shape (N, M, 2, 2)

        # HACK: GPytorch needs the interleaved matrix for the Multitask distribution
        # Reshape into (2N, 2M) interleaved matrix
        K_interleaved = K_blocks.permute(0, 2, 1, 3).reshape(2 * row_tensor.shape[0], 2 * column_tensor.shape[0])

        # Add this for Quantile Coverage Error (QCE) calculation
        if diag:
            return K_interleaved.diag()

        # NOTE: Lazify was replaced with to_linear_operator
        return to_linear_operator(K_interleaved)

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

        ### MEAN MODULE ###
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks = 2
            )

        ### COVARIANCE MODULE ###
        self.base_kernel = dfRBFKernel().to(device)
        # Wrap the base kernel in a scaling wrapper
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel)
        
        ### HYPERPARAMETERS INITIALIZATION ###
        # initialize hyperparameters by sampling from a uniform distribution over predefined ranges
        self.base_kernel.lengthscale = torch.empty(2, device = device).uniform_( * L_RANGE)
        self.covar_module.outputscale = torch.empty(1, device = device).uniform_( * SIGMA_F_RANGE)
        self.likelihood.noise = torch.empty(1, device = device).uniform_( * SIGMA_N_RANGE)

        ### CONSTRAINTS ###
        # add constraint to likelihood 1e-5 is the default
        self.base_kernel.raw_lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-5)
        self.covar_module.raw_outputscale_constraint = gpytorch.constraints.GreaterThan(1e-5)
        self.likelihood.raw_noise_constraint = gpytorch.constraints.GreaterThan(1e-5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        # NOTE: Specify x1 and x2 for the dfRBFKernel
        covar_x = self.covar_module.forward(x, x)
        # NOTE: Assure it is multitask
        # NOTE: The interleaved format is the default, but we specify it explicitly
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x, interleaved = True)

##############
### dfGPcm ###
##############

from configs import L_RANGE, SIGMA_F_RESIDUAL_MODEL_RANGE, SIGMA_N_RANGE
# NOTE: Initialise with a different SIGMA_F_RANGE for the residual model

class dfGPcm(gpytorch.models.ExactGP):
    # dfGP model with contant mean
    def __init__(self, train_x, train_y, likelihood, mean_vector):
        super().__init__(train_x, train_y, likelihood)

        ### MEAN MODULE ###
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks = 2
            )

        ### COVARIANCE MODULE ###
        self.base_kernel = dfRBFKernel().to(device)
        # Wrap the base kernel in a scaling wrapper
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel)
        
        ### MEAN INITIALIZATION ###
        self.mean_module.base_means[0].initialize(constant = mean_vector[0])
        self.mean_module.base_means[1].initialize(constant = mean_vector[1])
        
        ### HYPERPARAMETERS INITIALIZATION ###
        # initialize hyperparameters by sampling from a uniform distribution over predefined ranges
        self.base_kernel.lengthscale = torch.empty(2, device = device).uniform_( * L_RANGE)
        # NOTE: This is different from dfGP, because we model the residuals, which are smaller in magnitude
        # so we use a smaller range for the outputscale
        self.covar_module.outputscale = torch.empty(1, device = device).uniform_( * SIGMA_F_RESIDUAL_MODEL_RANGE)
        self.likelihood.noise = torch.empty(1, device = device).uniform_( * SIGMA_N_RANGE)

        ### CONSTRAINTS ###
        # add constraint to likelihood 1e-5 is the default
        self.base_kernel.raw_lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-5)
        self.covar_module.raw_outputscale_constraint = gpytorch.constraints.GreaterThan(1e-5)
        self.likelihood.raw_noise_constraint = gpytorch.constraints.GreaterThan(1e-5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        # NOTE: Specify x1 and x2 for the dfRBFKernel
        covar_x = self.covar_module.forward(x, x)
        # NOTE: Assure it is multitask
        # NOTE: The interleaved format is the default, but we specify it explicitly
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x, interleaved = True)

#############   
### dfNGP ###
#############

from configs import L_RANGE, SIGMA_F_RESIDUAL_MODEL_RANGE, SIGMA_N_RANGE

class dfNGP(gpytorch.models.ExactGP):
    # dfGP model with constant mean
    def __init__(self, train_x, train_y, likelihood):
        super(dfNGP, self).__init__(train_x, train_y, likelihood)

        ### CUSTOM MEAN MODULE ###
        self.mean_module = dfNN(
            input_dim = 2) # default hidden_dim = 32
        
        ### COVARIANCE MODULE ###
        self.base_kernel = dfRBFKernel().to(device)
        # Wrap the base kernel in a scaling wrapper
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel)

        ### HYPERPARAMETERS INITIALIZATION ###
        # initialize hyperparameters by sampling from a uniform distribution over predefined ranges
        self.base_kernel.lengthscale = torch.empty(2, device = device).uniform_( * L_RANGE)
        # NOTE: This is different from dfGP, because we model the residuals, which are smaller in magnitude
        # so we use a smaller range for the outputscale
        self.covar_module.outputscale = torch.empty(1, device = device).uniform_( * SIGMA_F_RESIDUAL_MODEL_RANGE)
        self.likelihood.noise = torch.empty(1, device = device).uniform_( * SIGMA_N_RANGE)

        ### CONSTRAINTS ###
        # add constraint to likelihood 1e-5 is the default
        self.base_kernel.raw_lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-5)
        self.covar_module.raw_outputscale_constraint = gpytorch.constraints.GreaterThan(1e-5)
        self.likelihood.raw_noise_constraint = gpytorch.constraints.GreaterThan(1e-5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        # NOTE: Specify x1 and x2 for the dfRBFKernel
        covar_x = self.covar_module.forward(x, x)
        # NOTE: Assure it is multitask
        # NOTE: The interleaved format is the default, but we specify it explicitly
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x, interleaved = True)

##########   
### GP ###
##########

from configs import SIGMA_N_RANGE, L_RANGE, SIGMA_F_RANGE, COVAR_OFFDIAGONAL_RANGE

class GP(gpytorch.models.ExactGP):
    # GP model with zero mean
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        ### MEAN MODULE ###
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks = 2
            )

        ### COVARIANCE MODULE ###
        # NOTE: We do not need a scaling wrapper here because the Multitask kernel already handles the scaling
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims = 2), 
            num_tasks = 2,
            rank = 1,
            )
        
        ### HYPERPARAMETERS INITIALIZATION ###
        self.likelihood.noise = torch.empty(1, device = device).uniform_( * SIGMA_N_RANGE)
        # NOTE: respect the dimensionalities used in gpytorch
        self.covar_module.data_covar_module.lengthscale = torch.empty([1, 2], device = device).uniform_( * L_RANGE)
        # Covariance between tasks (off-diagonal elements), can be negative
        # self.covar_module.task_covar_module.covar_factor = torch.empty([2, 1], device = device).uniform_( * COVAR_OFFDIAGONAL_RANGE)
        self.covar_module.task_covar_module.covar_factor.data.uniform_( * COVAR_OFFDIAGONAL_RANGE)
        # Independent variance for each task
        self.covar_module.task_covar_module.var = torch.empty(2, device = device).uniform_( * SIGMA_F_RANGE)

        ### CONSTRAINTS ###
        # Print:
        # likelihood.raw_noise torch.Size([1])
        # covar_module.task_covar_module.covar_factor torch.Size([2, 1])
        # covar_module.task_covar_module.raw_var torch.Size([2])
        # covar_module.data_covar_module.raw_lengthscale torch.Size([1, 2])

        # add constraint to likelihood 1e-4 is the default
        self.likelihood.raw_noise_constraint = gpytorch.constraints.GreaterThan(1e-5)
        self.covar_module.data_covar_module.raw_lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-5)
        # NOTE: covar factor can be negative or positive
        self.covar_module.task_covar_module.raw_var_constraint = gpytorch.constraints.GreaterThan(1e-5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        # Use default forward method of MultitaskKernel
        covar_x = self.covar_module(x)
        # NOTE: Assure it is multitask
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
