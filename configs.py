# Hyperparameters
PATIENCE = 200  # Stop after 100 epochs with no improvement (50 before)
MAX_NUM_EPOCHS = 2000  # Adjust based on training needs - check if it gets exhausted
NUM_RUNS = 10  # Number of training runs for metric evaluation

LEARNING_RATE = 0.0001

# higher LR works better for both, maybe try higher patience
PINN_LEARNING_RATE = 0.005 # back to regular lr
DFNN_LEARNING_RATE = 0.0001 # back to lr

GP_LEARNING_RATE = 0.01

# WEIGHT_DECAY is L2 regularisation, decay because it pulls weights towards 0
WEIGHT_DECAY = 1e-4 # don't want it too smooth 1e-5 for PINN was slighly worse
BATCH_SIZE = 32

# Discretization for test data
N_SIDE = 20
N_SIDE_VIS = 15

# Paths
DFNN_RESULTS_DIR = "results/dfNN"
DFNN_FULLMATRIX_RESULTS_DIR = "results/dfNN_fullmatrix"
HNN_RESULTS_DIR = "results/HNN"
PINN_RESULTS_DIR = "results/PINN"
DFGP_RESULTS_DIR = "results/dfGP"
GP_RESULTS_DIR = "results/GP"
DFNGP_RESULTS_DIR = "results/dfNGP"

# FOR PINN ONLY
W_PINN_DIV_WEIGHT = 0.3 # 0.5 before

# FOR GPs ONLY
SIGMA_N_RANGE = (0.02, 0.07) # Initialisation range
SIGMA_F_RANGE = (1.5, 2.0) # Initialisation range
L_RANGE = (0.3, 0.8) # Example range for each lengthscale

# For regular GP only
B_DIAGONAL_RANGE = SIGMA_F_RANGE  # Example range for B diagonal
B_OFFDIAGONAL_RANGE = (-0.2, 0.5)  # Example range for B off-diagonal

# we do not need as many
GP_MAX_NUM_EPOCHS = 1000

# noise parameter for training: independent Gaussian noise
STD_GAUSSIAN_NOISE = 0.02 # variance is 0.0004

