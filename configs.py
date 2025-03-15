# Hyperparameters
PATIENCE = 50  # Stop after 40 epochs with no improvement
MAX_NUM_EPOCHS = 2000  # Adjust based on training needs - check if it gets exhausted
NUM_RUNS = 10  # Number of training runs for metric evaluation

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32

# Discretization for test data
N_SIDE = 20

# Paths
DFNN_RESULTS_DIR = "results/dfNN"
PINN_RESULTS_DIR = "results/PINN"
DFGP_RESULTS_DIR = "results/DFGP"

# FOR PINN ONLY
W_PINN_DIV_WEIGHT = 0.5

# FOR GPS
SIGMA_F_RANGE = (1.5, 2.0)   # Example range for sigma_f
L_RANGE = (0.3, 0.8)         # Example range for each lengthscale
GP_LEARNING_RATE = 0.01

