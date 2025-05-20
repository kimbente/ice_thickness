################
### SIM DATA ###
################

# directories
GP_SIM_RESULTS_DIR = "results_sim/GP"
dfGP_SIM_RESULTS_DIR = "results_sim/dfGP"
dfNGP_SIM_RESULTS_DIR = "results_sim/dfNGP"
PINN_SIM_RESULTS_DIR = "results_sim/PINN"
dfNN_SIM_RESULTS_DIR = "results_sim/dfNN"

# learning rates
# NOTE: df is always smallcap
GP_SIM_LEARNING_RATE = 0.01
dfGP_SIM_LEARNING_RATE = 0.01 # NOTE: same as GP
dfNGP_SIM_LEARNING_RATE = 0.001
PINN_SIM_LEARNING_RATE = 0.005 # NOTE: larger lrs worked better for NN models
dfNN_SIM_LEARNING_RATE = 0.001 

# sim specific hyperparameters
# test grid resolution
N_SIDE = 20
N_SIDE_VIS = 15

#################
### REAL DATA ###
#################

# directories
GP_REAL_RESULTS_DIR = "results_real/GP"
dfGP_REAL_RESULTS_DIR = "results_real/dfGP"
dfNGP_REAL_RESULTS_DIR = "results_real/dfNGP"
PINN_REAL_RESULTS_DIR = "results_real/PINN"
dfNN_REAL_RESULTS_DIR = "results_real/dfNN"

# learning rates
GP_REAL_LEARNING_RATE = 0.01
dfGP_REAL_LEARNING_RATE = 0.01
dfNGP_REAL_LEARNING_RATE = 0.01
PINN_REAL_LEARNING_RATE = 0.001
dfNN_REAL_LEARNING_RATE = 0.001

################################
### TRAINING HYPERPARAMETERS ###
################################

NUM_RUNS = 10
MAX_NUM_EPOCHS = 2000

PATIENCE = 200  # Stop after {PATIENCE} epochs with no improvement
GP_PATIENCE = 50 # NOTE: GP convergence is more smooth so less patience is needed

# WEIGHT_DECAY is L2 regularisation, decay because it pulls weights towards 0
WEIGHT_DECAY = 1e-4 
BATCH_SIZE = 32

# FOR PINN ONLY
W_PINN_DIV_WEIGHT = 0.3 # 0.5 before

# Define initialisation ranges FOR GP MODELs
SIGMA_N_RANGE = (0.02, 0.07)
SIGMA_F_RANGE = (1.5, 2.0) 
L_RANGE = (0.3, 0.8) 

# For regular GP only
B_DIAGONAL_RANGE = SIGMA_F_RANGE  # Example range for B diagonal
B_OFFDIAGONAL_RANGE = (-0.2, 0.5)  # Example range for B off-diagonal

# noise parameter for training: independent Gaussian noise
STD_GAUSSIAN_NOISE = 0.02 # variance is 0.0004

"""
# TODO: remove this
# LEARNING_RATE = 0.0001 # NOTE: use model specific lrs

# Paths
DFNN_RESULTS_DIR = "results/dfNN"
DFNN_FULLMATRIX_RESULTS_DIR = "results/dfNN_fullmatrix"
HNN_RESULTS_DIR = "results/HNN"
PINN_RESULTS_DIR = "results/PINN"
PINN_DOMAIN_DIV_RESULTS_DIR = "results/PINN_domain_div"
DFGP_RESULTS_DIR = "results/dfGP"
GP_RESULTS_DIR = "results/GP"
GP_ONEL_RESULTS_DIR = "results/GP_one_l"
DFNGP_RESULTS_DIR = "results/dfNGP"
"""

