################
### SIM DATA ###
################

# directories (alphabetic order)
dfGP_SIM_RESULTS_DIR = "results_sim/dfGP"
dfGPcm_SIM_RESULTS_DIR = "results_sim/dfGPcm"
dfNGP_SIM_RESULTS_DIR = "results_sim/dfNGP"
dfNN_SIM_RESULTS_DIR = "results_sim/dfNN"
GP_SIM_RESULTS_DIR = "results_sim/GP"
PINN_SIM_RESULTS_DIR = "results_sim/PINN"

# learning rates (alphabetic order)
# NOTE: df is always smallcap.
# NOTE: We use a GP and a NN lr
dfGP_SIM_LEARNING_RATE = 0.01 
dfGPcm_SIM_LEARNING_RATE = 0.01
dfNGP_SIM_LEARNING_RATE = 0.01 # lr x 0.1 for NN mean function params
dfNN_SIM_LEARNING_RATE = 0.005
GP_SIM_LEARNING_RATE = 0.01
PINN_SIM_LEARNING_RATE = 0.005

# sim specific hyperparameters
# test grid resolution
N_SIDE = 20

#################
### REAL DATA ###
#################

# directories (alphabetic order)
dfGP_REAL_RESULTS_DIR = "results_real/dfGP"
dfGPcm_REAL_RESULTS_DIR = "results_real/dfGPcm"
dfNGP_REAL_RESULTS_DIR = "results_real/dfNGP"
dfNN_REAL_RESULTS_DIR = "results_real/dfNN"
GP_REAL_RESULTS_DIR = "results_real/GP"
PINN_REAL_RESULTS_DIR = "results_real/PINN"

# learning rates (alphabetic order)
# NOTE: GP lrs are 1/10th of sim lrs, because real data is more complex. NN lrs are the same.
dfGP_REAL_LEARNING_RATE = 0.001 # smaller for real data
dfGPcm_REAL_LEARNING_RATE = 0.001
dfNGP_REAL_LEARNING_RATE = 0.001 # lr x 0.1 for NN mean function params
dfNN_REAL_LEARNING_RATE = 0.001 # 0.005 worked, but could try slower makes it smoother
GP_REAL_LEARNING_RATE = 0.001 # needs to be smaller
PINN_REAL_LEARNING_RATE = 0.0001 # 0.005 was too fast

################################
### TRAINING HYPERPARAMETERS ###
################################

TRACK_EMISSIONS_BOOL = False

NUM_RUNS = 8 # previously 10
MAX_NUM_EPOCHS = 2000

PATIENCE = 100  # Stop after {PATIENCE} epochs with no improvement
GP_PATIENCE = 50 # NOTE: GP convergence is more smooth so less patience is needed

# WEIGHT_DECAY is L2 regularisation, decay because it pulls weights towards 0
WEIGHT_DECAY = 1e-4 # 0.0001
dfNN_SIM_WEIGHT_DECAY = 1e-2 # 0.01

BATCH_SIZE = 32

# PINN HYPERPARAM (SIM & REAL)
W_PINN_DIV_WEIGHT = 0.3

# FOR SIM
# Noise parameter for training: independent Gaussian noise to perturb inputs
# NOTE: This corresponds to a true noise variance of 0.0004
STD_GAUSSIAN_NOISE = 0.02

###################################
### REAL (df)GP HYPERPARAMETERS ###
###################################

# Scale input bacl to ~km
SCALE_INPUT_region_lower_byrd = 30
SCALE_INPUT_region_mid_byrd = 70
SCALE_INPUT_region_upper_byrd = 70

# NOTE: This corresponds to a l^2 range of (4.0, 25.0) (domain is [0, 100])
REAL_L_RANGE = (1.5, 4.0)

# NOTE: This corresponds to a sigma_n range of (,)
# REAL_NOISE_VAR_RANGE = (0.04, 0.07)
REAL_NOISE_VAR_RANGE = (0.02, 0.05) 

# REAL_OUTPUTSCALE_VAR_RANGE = (0.8, 1.8)
REAL_OUTPUTSCALE_VAR_RANGE = (1.0, 2.0)

##############################
### (df)GP HYPERPARAMETERS ###
##############################
# order: lengthscale, outputscale variance, noise variance

# HYPERPARAMETER 1: Range for lengthscale parameter (l) 
# NOTE: This corresponds to a l^2 range of (0.09, 0.64) (domain is [0, 1])
L_RANGE = (0.3, 0.8) 

# HYPERPARAMETER 2: Range for outputscale variance parameter (sigma_f^2)
# NOTE: This corresponds to a sigma_f range of (0.64, ~1.22)
OUTPUTSCALE_VAR_RANGE = (0.8, 1.5)
# formerly SIGMA_F_RANGE = (0.8, 1.5)

# NOTE: For residual models (i.e. models with non-zero mean function), we use a different range for the outputscale variance, acknowledging that the residuals are smaller than the original data.
OUTPUTSCALE_VAR_RESIDUAL_MODEL_RANGE = (0.1, 0.6)
# Formerly SIGMA_F_RESIDUAL_MODEL_RANGE = (0.1, 0.6)

# For regular GP only, we scale for each task
# NOTE: The multitask GP is parameterised via a covariance factor F, which is used to construct the covariance matrix B together with a TASK Variance D.
# B = (FF^T + D), where D is a diagonal matrix and F is the covar_factor
TASK_COVAR_FACTOR_RANGE = (-0.2, 0.5) 
# Formerly COVAR_OFFDIAGONAL_RANGE = (-0.2, 0.5) 

# Define initialisation ranges FOR GP MODELs
# HYPERPARAMETER 3: Range for noise variance parameter (sigma_n^2)
# NOTE: This corresponds to a sigma_n range of (0.01, 0.07)
NOISE_VAR_RANGE = (0.0001, 0.0049)
# formerly SIGMA_N_RANGE = (0.02, 0.07)
