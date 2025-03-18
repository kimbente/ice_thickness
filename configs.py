# Hyperparameters
PATIENCE = 50  # Stop after 40 epochs with no improvement
MAX_NUM_EPOCHS = 2000  # Adjust based on training needs - check if it gets exhausted
NUM_RUNS = 10  # Number of training runs for metric evaluation

LEARNING_RATE = 0.0001
# higher LR works better for both, maybe try higher patiene
PINN_LEARNING_RATE = 0.001
DFNN_LEARNING_RATE = 0.001

# WEIGHT_DECAY = 1e-4
WEIGHT_DECAY = 0.01  # 1e-2
BATCH_SIZE = 32

# Discretization for test data
N_SIDE = 20

# Paths
DFNN_RESULTS_DIR = "results/dfNN"
PINN_RESULTS_DIR = "results/PINN"
DFGP_RESULTS_DIR = "results/dfGP"
GP_RESULTS_DIR = "results/GP"
DFGPDFNN_RESULTS_DIR = "results/dfGPdfNN"

# FOR PINN ONLY
W_PINN_DIV_WEIGHT = 0.5

# FOR GPS
SIGMA_F_RANGE = (1.5, 2.0)   # Example range for sigma_f
# For regular GP only
B_DIAGONAL_RANGE = SIGMA_F_RANGE  # Example range for B diagonal
B_OFFDIAGONAL_RANGE = (- 0.2, 0.5)  # Example range for B off-diagonal

L_RANGE = (0.3, 0.8)         # Example range for each lengthscale
GP_LEARNING_RATE = 0.01

GP_MAX_NUM_EPOCHS = 1000

