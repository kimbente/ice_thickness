model_name = "dfNGP"
from gpytorch_models import dfNGP

# import configs to we can access the hypers with getattr
import configs
from configs import PATIENCE, MAX_NUM_EPOCHS, NUM_RUNS, WEIGHT_DECAY
from configs import TRACK_EMISSIONS_BOOL
from configs import SCALE_INPUT_region_lower_byrd, SCALE_INPUT_region_mid_byrd, SCALE_INPUT_region_upper_byrd
from configs import REAL_L_RANGE, REAL_NOISE_VAR_RANGE, REAL_OUTPUTSCALE_VAR_RANGE

SCALE_INPUT = {
    "region_lower_byrd": SCALE_INPUT_region_lower_byrd,
    "region_mid_byrd": SCALE_INPUT_region_mid_byrd,
    "region_upper_byrd": SCALE_INPUT_region_upper_byrd,
}

# Reiterating import for visibility
MAX_NUM_EPOCHS = MAX_NUM_EPOCHS
NUM_RUNS = NUM_RUNS
NUM_RUNS = 1
WEIGHT_DECAY = WEIGHT_DECAY
PATIENCE = PATIENCE

# assign model-specific variable
MODEL_LEARNING_RATE = getattr(configs, f"{model_name}_REAL_LEARNING_RATE")
MODEL_REAL_RESULTS_DIR = getattr(configs, f"{model_name}_REAL_RESULTS_DIR")
import os
os.makedirs(MODEL_REAL_RESULTS_DIR, exist_ok = True)

# basics
import pandas as pd
import torch
import gpytorch

# universals 
from metrics import compute_divergence_field, quantile_coverage_error_2d
from utils import set_seed, make_grid
import gc
import warnings
set_seed(42)

# setting device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# overwrite if needed: # device = 'cpu'
print('Using device:', device)
print()

### START TIMING ###
import time
start_time = time.time()  # Start timing after imports

### START TRACKING EXPERIMENT EMISSIONS ###
if TRACK_EMISSIONS_BOOL:
    from codecarbon import EmissionsTracker
    tracker = EmissionsTracker(project_name = "dfNGP_real_experiments", output_dir = MODEL_REAL_RESULTS_DIR)
    tracker.start()

#############################
### LOOP 1 - over REGIONS ###
#############################

for region_name in ["region_lower_byrd", "region_mid_byrd", "region_upper_byrd"]:
    SCALE_DOMAIN = SCALE_INPUT[region_name]

    print(f"\nTraining for {region_name.upper()}...")

    # Store metrics for the current region (used for *metrics_summary* report and *metrics_per_run*)
    region_results = []

    ##########################################
    ### x_train & y_train, x_test & x_test ###
    ##########################################

    # define paths based on region_name
    path_to_training_tensor = "data/real_data/" + region_name + "_train_tensor.pt"
    path_to_test_tensor = "data/real_data/" + region_name + "_test_tensor.pt"

    # load and tranpose to have rows as points
    train = torch.load(path_to_training_tensor, weights_only = False).T 
    test = torch.load(path_to_test_tensor, weights_only = False).T

    # The train and test tensors have the following columns:
    # [:, 0] = x
    # [:, 1] = y
    # [:, 2] = surface elevation (s)
    # [:, 3] = ice flux in x direction (u)
    # [:, 4] = ice flux in y direction (v)
    # [:, 5] = ice flux error in x direction (u_err)
    # [:, 6] = ice flux error in y direction (v_err)
    # [:, 7] = source age

    # train
    x_train = train[:, [0, 1]].to(device)
    y_train = train[:, [3, 4]].to(device)

    # test
    x_test = test[:, [0, 1]].to(device)
    y_test = test[:, [3, 4]].to(device)

    # HACK: Scaling helps with numerical stability
    # Units are not in km 
    x_test = x_test * SCALE_DOMAIN
    x_train = x_train * SCALE_DOMAIN

    # NOTE: Here we estimate the noise variance 

    # Print train details
    print(f"=== {region_name.upper()} ===")
    print(f"Training inputs shape: {x_train.shape}")
    print(f"Training observations shape: {y_train.shape}")
    print(f"Training inputs dtype: {x_train.dtype}")
    print()

    # Print test details
    print(f"=== {region_name.upper()} ===")
    print(f"Test inputs shape: {x_test.shape}")
    print(f"Test observations shape: {y_test.shape}")
    print(f"Test inputs dtype: {x_test.dtype}")
    print()

    ##################################
    ### LOOP 2 - over training run ###
    ##################################

    # NOTE: GPs don't train on batches, use full data

    for run in range(NUM_RUNS):

        print(f"\n--- Training Run {run + 1}/{NUM_RUNS} ---")

        # Initialise the likelihood for the GP model (estimates noise)
        # NOTE: we use a multitask likelihood for the dfNGP model but with a global noise term
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks = 2,
            has_global_noise = True, 
            has_task_noise = False, # HACK: This still needs to be manually turned off
            ).to(device)

        # NOTE: This was needed
        x_train = x_train.clone().detach().requires_grad_(True)

        model = dfNGP(
            x_train,
            y_train, 
            likelihood
            ).to(device)
        
        # Initialise hypers as usual 
        # NOTE: Alternative is to start with best hypers from previously trained model
        # Overwrite default lengthscale hyperparameter initialisation because we have a different input scale.
        model.base_kernel.lengthscale = torch.empty([1, 2], device = device).uniform_( * REAL_L_RANGE)
        # Overwrite default outputscale variance initialisation.
        model.covar_module.outputscale = torch.empty(1, device = device).uniform_( * REAL_OUTPUTSCALE_VAR_RANGE)
        # Overwrite default noise variance initialisation because this is real noisy data.
        model.likelihood.noise = torch.empty(1, device = device).uniform_( * REAL_NOISE_VAR_RANGE)

        # NOTE: This part is different from dfGP
        optimizer = torch.optim.AdamW([
            {"params": model.mean_module.parameters(), 
             "weight_decay": WEIGHT_DECAY, "lr": MODEL_LEARNING_RATE * 0.2},
            {"params": list(model.covar_module.parameters()) + list(model.likelihood.parameters()), 
             "weight_decay":  WEIGHT_DECAY, "lr": MODEL_LEARNING_RATE},
            ])
        
        # Use ExactMarginalLogLikelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Early stopping variables
        best_loss = float('inf')
        # counter starts at 0
        epochs_no_improve = 0

        ############################
        ### LOOP 3 - over EPOCHS ###
        ############################
        print("\nStart Training")

        for epoch in range(MAX_NUM_EPOCHS):

            # Set to train
            model.train()
            likelihood.train()

            # Do a step
            optimizer.zero_grad()

            x_train_grid = x_train_grid.clone().detach().requires_grad_(True)
            train_pred_dist = model(x_train_grid.to(device))
            loss = - mll(train_pred_dist, y_train_grid.to(device))  # negative marginal log likelihood
            loss.backward()
            optimizer.step()
        