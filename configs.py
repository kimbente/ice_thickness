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
dfNN_REAL_LEARNING_RATE = 0.005 
GP_REAL_LEARNING_RATE = 0.001 # needs to be smaller
PINN_REAL_LEARNING_RATE = 0.005

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

# FOR PINN ONLY
W_PINN_DIV_WEIGHT = 0.3

# Define initialisation ranges FOR GP MODELs
SIGMA_N_RANGE = (0.02, 0.07)
# SIGMA_F_RANGE = (1.5, 2.0) 
SIGMA_F_RANGE = (0.8, 1.5)
SIGMA_F_RESIDUAL_MODEL_RANGE = (0.1, 0.6) # for dfNGP we model the residuals, so we need a different sigma_f range
L_RANGE = (0.3, 0.8) 

# For regular GP only
COVAR_OFFDIAGONAL_RANGE = (-0.2, 0.5) 

# FOR SIM
# Noise parameter for training: independent Gaussian noise
STD_GAUSSIAN_NOISE = 0.02 # variance is 0.0004