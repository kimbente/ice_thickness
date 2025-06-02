################
### SIM DATA ###
################

# directories (alphabetic order)
dfGP_SIM_RESULTS_DIR = "results_sim/dfGP"
dfNGP_SIM_RESULTS_DIR = "results_sim/dfNGP"
dfNN_SIM_RESULTS_DIR = "results_sim/dfNN"
GP_SIM_RESULTS_DIR = "results_sim/GP"
PINN_SIM_RESULTS_DIR = "results_sim/PINN"

# learning rates (alphabetic order)
# NOTE: df is always smallcap
dfGP_SIM_LEARNING_RATE = 0.01 # NOTE: same as GP
dfNGP_SIM_LEARNING_RATE = 0.001
dfNN_SIM_LEARNING_RATE = 0.005 # NOTE: We run it with 0.001 before but that was too small. This makes quite a difference
GP_SIM_LEARNING_RATE = 0.01
PINN_SIM_LEARNING_RATE = 0.005 # NOTE: larger lrs worked better for NN models

# sim specific hyperparameters
# test grid resolution
N_SIDE = 20
N_SIDE_VIS = 15

#################
### REAL DATA ###
#################

# directories (alphabetic order)
dfGP_REAL_RESULTS_DIR = "results_real/dfGP"
dfNGP_REAL_RESULTS_DIR = "results_real/dfNGP"
dfNN_REAL_RESULTS_DIR = "results_real/dfNN"
GP_REAL_RESULTS_DIR = "results_real/GP"
PINN_REAL_RESULTS_DIR = "results_real/PINN"

# learning rates (alphabetic order)
dfGP_REAL_LEARNING_RATE = 0.01
dfNGP_REAL_LEARNING_RATE = 0.01
dfNN_REAL_LEARNING_RATE = 0.001
GP_REAL_LEARNING_RATE = 0.01
PINN_REAL_LEARNING_RATE = 0.001

################################
### TRAINING HYPERPARAMETERS ###
################################

NUM_RUNS = 10
MAX_NUM_EPOCHS = 2000

PATIENCE = 100  # Stop after {PATIENCE} epochs with no improvement
GP_PATIENCE = 50 # NOTE: GP convergence is more smooth so less patience is needed

# WEIGHT_DECAY is L2 regularisation, decay because it pulls weights towards 0
WEIGHT_DECAY = 1e-4 # 0.0001
BATCH_SIZE = 32

# FOR PINN ONLY
W_PINN_DIV_WEIGHT = 0.3

# Define initialisation ranges FOR GP MODELs
SIGMA_N_RANGE = (0.02, 0.07)
SIGMA_F_RANGE = (1.5, 2.0) 
L_RANGE = (0.3, 0.8) 

# For regular GP only
B_DIAGONAL_RANGE = SIGMA_F_RANGE
B_OFFDIAGONAL_RANGE = (-0.2, 0.5) 

# Noise parameter for training: independent Gaussian noise
STD_GAUSSIAN_NOISE = 0.02 # variance is 0.0004