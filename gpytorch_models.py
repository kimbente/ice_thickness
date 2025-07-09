import torch
import gpytorch

# For the kernel
import linear_operator
from linear_operator.operators import KroneckerProductLinearOperator
from gpytorch.kernels.rbf_kernel import postprocess_rbf, RBFKernel

from configs import L_RANGE, OUTPUTSCALE_VAR_RANGE, OUTPUTSCALE_VAR_RESIDUAL_MODEL_RANGE, NOISE_VAR_RANGE

# setting device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##################
### df KERNEL ####
##################

class dfRBFKernel(RBFKernel):
    r"""
    Computes a divergence-free RBF covariance matrix of the RBF kernel that models the covariance
    for inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`. The code is heavily based on the Hessian component of the RBF grad kernel, also inheriting from Gpytorch's RBF kernel (see source code here: https://docs.gpytorch.ai/en/stable/_modules/gpytorch/kernels/rbf_kernel_grad.html#RBFKernelGrad) to enable computational speed ups.

    Notes: 
    - This only works for 2D currently. 
    - Uses self.lengthscale object.

    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    :param ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `d` if x1 is a `n x d` matrix. (Default: `None`.)
    :param batch_shape: Set this if you want a separate lengthscale for each batch of input
        data. It should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf x1` is
        a :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    :param lengthscale_prior: Set this if you want to apply a prior to the
        lengthscale parameter. (Default: `None`)
    :param lengthscale_constraint: Set this if you want to apply a constraint
        to the lengthscale parameter. (Default: `Positive`.)
    :param eps: The minimum value that the lengthscale can take (prevents
        divide by zero errors). (Default: `1e-6`.)

    :ivar torch.Tensor lengthscale: The lengthscale parameter. Size/shape of parameter depends on the
        ard_num_dims and batch_shape arguments.

    Example:
        >>> x = torch.randn(10, 2)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.dfRBFKernel())
        >>> covar = covar_module(x)  # Output: LinearOperator of size (20 x 20), where 20 = n * d
    """

    def forward(self, x1, x2, diag = False, **params):

        ### STEP 1: EXTRACT BATCHES & DIMS ###

        # BATCHES
        # NOTE: We write this in a batch compatible way, using negative indexing (from the right)
        # batch_shape is torch.Size([]) if only 2D
        batch_shape = x1.shape[:-2]
        # usually zero, one, or two
        n_batch_dims = len(batch_shape)

        # DIMS
        N, d = x1.shape[-2:]
        # d must be same
        M = x2.shape[-2]

        # We save compute when diag = True by only computing what is needed
        if not diag:

            ### STEP 2: DIRECTIONAL SCALED DIFFERENCES ###
            # Scale the inputs by the lengthscale(s) (e.g. two lengthscales if ard_num_dims = 2)
            # HACK: Applying div before subtracting increases stability
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)

            # Broadcast and compute directional (scaled) differences (..., N, M, d)
            directional_scaled_diffs = x1_.view(*batch_shape, N, 1, d) - x2_.view(*batch_shape, 1, M, d)
            # Divide by lengthscales again to get directional_scaled_diffs: (x1 - x2) / ℓ^2
            # lengthscale e.g. shape (1, 2) (unsqueeze to broadcast)
            directional_scaled_diffs = directional_scaled_diffs / self.lengthscale.unsqueeze(-2)

            # HACK: Flip the last axis to have d2, d1 ordering. The off-diagonals thus remain the same while the diagonals are flipped. This is new for the dfRBF kernel.
            directional_scaled_diffs = torch.flip(directional_scaled_diffs, dims = [-1])

            # (..., N, d, M): swops last two dims
            directional_scaled_diffs = torch.transpose(directional_scaled_diffs, -1, -2).contiguous()

            # torch.Size([..., N, d * M]) where the first M indices are d1 and the latter M indices are d2 for 2d
            # d1 diffs are the left block, d2 diffs are the right block
            directional_scaled_diffs_N_rows_wide = directional_scaled_diffs.view(*batch_shape, N, d * M)
            
            # transpose swop N and M so makes torch.Size([..., M, d, N])
            # reshape torch.Size([..., M, d * N])
            directional_scaled_diffs_M_rows_wide = directional_scaled_diffs.transpose(-1, -3).reshape(*batch_shape, M, d * N)
            # d1 diffs are the upper block, d2 diffs are the lower block
            directional_scaled_diffs_2N_rows_tall = directional_scaled_diffs_M_rows_wide.transpose(-1, -2)

            # 1) Kernel block
            # NOTE: as an instance of the RBF kernel, we can use the covar_dist method to compute the squared distance;
            # This uses our lengthscale.
            diff = self.covar_dist(x1_, x2_, square_dist = True, **params)
            # torch.Size([..., N, M])
            K_rbf = postprocess_rbf(diff)

            ### STEP 4: BLOCKWISE ASSIGNMENTS ###
            # torch.Size([..., N * d, M * d])
            # First term: Last axis stays the same, second last axis (rows) is repeated
            directional_scaled_diffs_two_rows = directional_scaled_diffs_N_rows_wide.repeat([*([1] * n_batch_dims), d, 1]) 
            # Second term: Last axis (columns) is repeated, second last axis (rows) stays the same
            directional_scaled_diffs_two_columns = directional_scaled_diffs_2N_rows_tall.repeat([*([1] * (n_batch_dims + 1)), d])

            # Upper left is r_1**2/l_1**4
            # Upper right and lower left are r_1 * r_2 / (l_1**2 * l_2**2)
            # Lower right is r_2**2 / l_2**4
            directional_scaled_diffs_product_block = directional_scaled_diffs_two_rows * directional_scaled_diffs_two_columns

            ### STEP 5: SIGN BLOCK ###
            # Create sign block to negate off-diagonal blocks
            sign_block = KroneckerProductLinearOperator(
                torch.tensor([[1., -1.], [-1., 1.]], device = x1.device, dtype = x1.dtype).repeat(*batch_shape, 1, 1),
                torch.ones(N, M, device = x1.device, dtype = x1.dtype).repeat(*batch_shape, 1, 1)
            )

            directional_scaled_diffs_product_block = sign_block * directional_scaled_diffs_product_block

            ### STEP 6: KRONECKER PRODUCT BLOCK ###
            # torch.Size([..., N * d, M * d])
            kp = KroneckerProductLinearOperator(
                        # Upper left block is (1/l1^2)
                        # Lower right block is (1/l2^2)
                        # Upper right and lower left blocks are (0)
                        # HACK: Flip the last axis to match the dfRBF ordering. This is another change from the RBF grad kernel.
                        torch.eye(d, d, device = x1.device, dtype = x1.dtype).repeat(*batch_shape, 1, 1) / self.lengthscale.flip(-1).pow(2),
                        # Expand
                        torch.ones(N, M, device = x1.device, dtype = x1.dtype).repeat(*batch_shape, 1, 1),
                    )
            
            ### STEP 6: SUBTRACT BLOCKS ###
            # Upper left block is (1/l1^2 - product)
            # Lower right block is (1/l2^2 - product)
            # Upper right and lower left blocks are (0 - product), resulting in the negative product
            subtracted_blocks = kp.to_dense() - directional_scaled_diffs_product_block
            # repeat K_rbf d * d times
            K = subtracted_blocks * K_rbf.repeat([*([1] * n_batch_dims), d, d])

            # Symmetrize for stability (Not for K_train_test of course)
            if N == M and torch.eq(x1, x2).all():
                K = 0.5 * (K.transpose(-1, -2) + K)

            # Apply a perfect shuffle permutation to match the MutiTask ordering
            pi1 = torch.arange(N * d).view(d, N).t().reshape((N * d))
            pi2 = torch.arange(M * d).view(d, M).t().reshape((M * d))
            K = K[..., pi1, :][..., :, pi2]

            return K

        # If diag = True, we only compute the diagonal elements
        else:
            if not (N == M and torch.eq(x1, x2).all()):
                raise RuntimeError("diag = True only works when x1 == x2")
            
            # NOTE: We have to flip the last axis of lengthscale to match the dfRBF ordering.
            grad_diag = torch.ones(*batch_shape, M, d, device = x1.device, dtype = x1.dtype) / self.lengthscale.flip(-1).pow(2)
            grad_diag = grad_diag.transpose(-1, -2).reshape(*batch_shape, M * d)

            # The permutation indices for the diagonal elements to match interleaved format in Multitask settinga
            pi = torch.arange(M * d).view(d, M).t().reshape((M * d))
            return grad_diag[..., pi]

    def num_outputs_per_input(self, x1, x2):
        return x1.size(-1)

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

from configs import L_RANGE, OUTPUTSCALE_VAR_RANGE, NOISE_VAR_RANGE

class dfGP(gpytorch.models.ExactGP):
    # dfGP model with zero mean
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        ### MEAN MODULE ###
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks = 2
            )

        ### COVARIANCE MODULE ###
        # self.base_kernel = dfRBFKernel_linop_afterthought(num_tasks = 2).to(device)
        # self.base_kernel = dfRBFKernel_kronecker(num_tasks = 2).to(device)
        # NOTE: We need to initialise it with ard_num_dims = 2  if we want to use a 2D lengthscale
        self.base_kernel = dfRBFKernel(ard_num_dims = 2).to(device)
        
        # Wrap the base kernel in a scaling wrapper
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel)
        
        self.covar_module.num_tasks = 2  # Ensure the covariance module is aware of the number of tasks)
        
        ### HYPERPARAMETERS INITIALIZATION ###
        # initialize hyperparameters by sampling from a uniform distribution over predefined ranges
        # NOTE: Models are initialised for sim experiments here. We overwrite the initialisations in the real data experiments.
        self.base_kernel.lengthscale = torch.empty(2, device = device).uniform_( * L_RANGE)
        # NOTE: The outputscale in gpytorch denotes σ², the outputscale variance, not σ
        # See https://docs.gpytorch.ai/en/latest/kernels.html#scalekernel
        self.covar_module.outputscale = torch.empty(1, device = device).uniform_( * OUTPUTSCALE_VAR_RANGE)
        # NOTE: Noise in gpytorch denotes σ², the noise variance, not σ
        # See https://docs.gpytorch.ai/en/latest/likelihoods.html#gaussianlikelihood 
        self.likelihood.noise = torch.empty(1, device = device).uniform_( * NOISE_VAR_RANGE)

        ### CONSTRAINTS ###
        self.base_kernel.raw_lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-5)
        self.covar_module.raw_outputscale_constraint = gpytorch.constraints.GreaterThan(1e-5)
        self.likelihood.raw_noise_constraint = gpytorch.constraints.GreaterThan(1e-5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        # Forward looks the most symmetric but huge negative values
        covar_x = self.covar_module(x)
        # NOTE: Assure it is multitask
        # NOTE: The interleaved = True format is the default, but we specify it explicitly
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

##############
### dfGPcm ###
##############

from configs import L_RANGE, OUTPUTSCALE_VAR_RESIDUAL_MODEL_RANGE, NOISE_VAR_RANGE
# NOTE: Initialise with a different OUTPUTSCALE_VAR_RANGE for the residual model

class dfGPcm(gpytorch.models.ExactGP):
    # dfGP model with contant mean
    def __init__(self, train_x, train_y, likelihood, mean_vector):
        super().__init__(train_x, train_y, likelihood)

        ### MEAN MODULE ###
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks = 2
            )

        ### COVARIANCE MODULE ###
        self.base_kernel = dfRBFKernel(ard_num_dims = 2).to(device)
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
        self.covar_module.outputscale = torch.empty(1, device = device).uniform_( * OUTPUTSCALE_VAR_RESIDUAL_MODEL_RANGE)
        self.likelihood.noise = torch.empty(1, device = device).uniform_( * NOISE_VAR_RANGE)

        ### CONSTRAINTS ###
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

from configs import L_RANGE, OUTPUTSCALE_VAR_RESIDUAL_MODEL_RANGE, NOISE_VAR_RANGE

class dfNGP(gpytorch.models.ExactGP):
    # dfGP model with constant mean
    def __init__(self, train_x, train_y, likelihood):
        super(dfNGP, self).__init__(train_x, train_y, likelihood)

        ### CUSTOM MEAN MODULE ###
        self.mean_module = dfNN(
            input_dim = 2) # default hidden_dim = 32
        
        ### COVARIANCE MODULE ###
        self.base_kernel = dfRBFKernel(ard_num_dims = 2).to(device)
        # Wrap the base kernel in a scaling wrapper
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel)

        ### HYPERPARAMETERS INITIALIZATION ###
        # initialize hyperparameters by sampling from a uniform distribution over predefined ranges
        self.base_kernel.lengthscale = torch.empty(2, device = device).uniform_( * L_RANGE)
        # NOTE: This is different from dfGP, because we model the residuals, which are smaller in magnitude
        # so we use a smaller range for the outputscale
        self.covar_module.outputscale = torch.empty(1, device = device).uniform_( * OUTPUTSCALE_VAR_RESIDUAL_MODEL_RANGE)
        self.likelihood.noise = torch.empty(1, device = device).uniform_( * NOISE_VAR_RANGE)

        ### CONSTRAINTS ###
        # add constraint to likelihood 1e-5 is the default
        self.base_kernel.raw_lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-5)
        self.covar_module.raw_outputscale_constraint = gpytorch.constraints.GreaterThan(1e-5)
        self.likelihood.raw_noise_constraint = gpytorch.constraints.GreaterThan(1e-5)

    def forward(self, x):
        # print("requires_grad?", x.requires_grad)
        mean_x = self.mean_module(x)  # Ensure gradients are computed for the mean
        # NOTE: Specify x1 and x2 for the dfRBFKernel
        covar_x = self.covar_module.forward(x, x)
        # NOTE: Assure it is multitask
        # NOTE: The interleaved format is the default, but we specify it explicitly
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x, interleaved = True)

##########   
### GP ###
##########

from configs import NOISE_VAR_RANGE, L_RANGE, OUTPUTSCALE_VAR_RANGE, TASK_COVAR_FACTOR_RANGE

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
        # NOTE: respect the dimensionalities used in gpytorch
        self.covar_module.data_covar_module.lengthscale = torch.empty([1, 2], device = device).uniform_( * L_RANGE)
        # Covariance between tasks (off-diagonal elements), can be negative
    
        # The Task variance-covarince matrix B is parameterised via a covariance factor F, which is used to construct the covariance matrix B together with a task variance D.
        # B = (FF^T + D), where D is a diagonal matrix and F is the covar_factor
        # See: https://github.com/cornellius-gp/gpytorch/discussions/2500#discussioncomment-8895555
        self.covar_module.task_covar_module.covar_factor.data.uniform_( * TASK_COVAR_FACTOR_RANGE)
        # Independent variance for each task. We use the same range for both tasks
        self.covar_module.task_covar_module.var = torch.empty(2, device = device).uniform_( * OUTPUTSCALE_VAR_RANGE)

        # NOTE: In Gpytorch noise is sigma**2, so the variance
        self.likelihood.noise = torch.empty(1, device = device).uniform_( * NOISE_VAR_RANGE)

        ### CONSTRAINTS ###

        self.covar_module.data_covar_module.raw_lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-5)
        # NOTE: covar factor can be negative or positive
        self.covar_module.task_covar_module.raw_var_constraint = gpytorch.constraints.GreaterThan(1e-5)
        self.likelihood.raw_noise_constraint = gpytorch.constraints.GreaterThan(1e-5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        # Use default forward method of MultitaskKernel
        covar_x = self.covar_module(x)
        # NOTE: Assure it is multitask
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
