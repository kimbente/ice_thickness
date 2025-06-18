################
### SIM DATA ###
################

# directories (alphabetic order)
dfGP_SIM_RESULTS_DIR = "results_sim/dfGP"
dfNGP_SIM_RESULTS_DIR = "results_sim/dfNGP"
dfNN_SIM_RESULTS_DIR = "results_sim/dfNN"
GP_SIM_RESULTS_DIR = "results_sim/GP"
PINN_SIM_RESULTS_DIR = "results_sim/PINN"

dfGP2_SIM_RESULTS_DIR = "results_sim/dfGP2"
dfGPcm_SIM_RESULTS_DIR = "results_sim/dfGPcm"

# learning rates (alphabetic order)
# NOTE: df is always smallcap.
# NOTE: We use a GP and a NN lr
dfGP_SIM_LEARNING_RATE = 0.01 
dfNGP_SIM_LEARNING_RATE = 0.01 # lr x 0.1 for NN mean function params
dfNN_SIM_LEARNING_RATE = 0.005
GP_SIM_LEARNING_RATE = 0.01
PINN_SIM_LEARNING_RATE = 0.005

dfGP2_SIM_LEARNING_RATE = 0.01
dfGPcm_SIM_LEARNING_RATE = 0.01

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

dfGP2_REAL_RESULTS_DIR = "results_real/dfGP2"
dfGPcm_REAL_RESULTS_DIR = "results_real/dfGPcm"

# learning rates (alphabetic order)
# NOTE: GP lrs are 1/10th of sim lrs, because real data is more complex. NN lrs are the same.
dfGP_REAL_LEARNING_RATE = 0.001 # smaller for real data
dfNGP_REAL_LEARNING_RATE = 0.001 # lr x 0.1 for NN mean function params
dfNN_REAL_LEARNING_RATE = 0.005 
GP_REAL_LEARNING_RATE = 0.001 # needs to be smaller
PINN_REAL_LEARNING_RATE = 0.005

dfGP2_REAL_LEARNING_RATE = 0.001
dfGPcm_REAL_LEARNING_RATE = 0.001

################################
### TRAINING HYPERPARAMETERS ###
################################

TRACK_EMISSIONS_BOOL = False

NUM_RUNS = 1 # 10
MAX_NUM_EPOCHS = 2000

PATIENCE = 100  # Stop after {PATIENCE} epochs with no improvement
GP_PATIENCE = 50 # NOTE: GP convergence is more smooth so less patience is needed

# WEIGHT_DECAY is L2 regularisation, decay because it pulls weights towards 0
WEIGHT_DECAY = 1e-4 # 0.0001
dfNN_SIM_WEIGHT_DECAY = 1e-2 # 0.01

BATCH_SIZE = 32

# FOR PINN ONLY
W_PINN_DIV_WEIGHT = 0.3

# Define initialisation ranges FOR GP MODELs
SIGMA_N_RANGE = (0.02, 0.07)
# SIGMA_F_RANGE = (1.5, 2.0) 
SIGMA_F_RANGE = (0.8, 1.5)
SIGMA_F_FIXED_RESIDUAL_MODEL_RANGE = (0.1, 0.6)
SIGMA_F_RESIDUAL_MODEL_RANGE = (0.1, 0.4) # for dfNGP we model the residuals, so we need a different sigma_f range
L_RANGE = (0.3, 0.8) 

# For regular GP only
B_DIAGONAL_RANGE = SIGMA_F_RANGE
B_OFFDIAGONAL_RANGE = (-0.2, 0.5) 

# FOR SIM
# Noise parameter for training: independent Gaussian noise
STD_GAUSSIAN_NOISE = 0.02 # variance is 0.0004