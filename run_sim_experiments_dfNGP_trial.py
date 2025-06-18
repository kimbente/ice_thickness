# SIMULATED DATA EXPERIMENTS
# # RUN WITH python run_sim_experiments_dfNGPcm.py
# 
#       ooooooooooooooooooooooooooooooooooooo
#      8                                .d88
#      8  oooooooooooooooooooooooooooood8888
#      8  8888888888888888888888888P"   8888    oooooooooooooooo
#      8  8888888888888888888888P"      8888    8              8
#      8  8888888888888888888P"         8888    8             d8
#      8  8888888888888888P"            8888    8            d88
#      8  8888888888888P"               8888    8           d888
#      8  8888888888P"                  8888    8          d8888
#      8  8888888P"                     8888    8         d88888
#      8  8888P"                        8888    8        d888888
#      8  8888oooooooooooooooooooooocgmm8888    8       d8888888
#      8 .od88888888888888888888888888888888    8      d88888888
#      8888888888888888888888888888888888888    8     d888888888
#                                               8    d8888888888
#         ooooooooooooooooooooooooooooooo       8   d88888888888
#        d                       ...oood8b      8  d888888888888
#       d              ...oood888888888888b     8 d8888888888888
#      d     ...oood88888888888888888888888b    8d88888888888888
#     dood8888888888888888888888888888888888b
#
#
# This artwork is a visual reminder that this script is for the sim experiments.

model_name = "dfNGP"
from gpytorch_models_old import dfNGP

# import configs to we can access the hypers with getattr
import configs
from configs import PATIENCE, MAX_NUM_EPOCHS, NUM_RUNS, WEIGHT_DECAY
# also import x_test grid size and std noise for training data
from configs import N_SIDE, STD_GAUSSIAN_NOISE
from configs import TRACK_EMISSIONS_BOOL

# Reiterating import for visibility
MAX_NUM_EPOCHS = MAX_NUM_EPOCHS
NUM_RUNS = NUM_RUNS
NUM_RUNS = 1
WEIGHT_DECAY = WEIGHT_DECAY
PATIENCE = PATIENCE

# assign model-specific variable
MODEL_LEARNING_RATE = getattr(configs, f"{model_name}_SIM_LEARNING_RATE")
MODEL_SIM_RESULTS_DIR = getattr(configs, f"{model_name}_SIM_RESULTS_DIR")
import os
os.makedirs(MODEL_SIM_RESULTS_DIR, exist_ok = True)

# basics
import pandas as pd
import torch
import gpytorch

# universals 
from metrics import compute_divergence_field
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
    tracker = EmissionsTracker(project_name = "dfGP_simulation_experiments", output_dir = MODEL_SIM_RESULTS_DIR)
    tracker.start()

### SIMULATION ###
# Import all simulation functions
from simulate import (
    simulate_detailed_branching,
    simulate_detailed_curve,
    simulate_detailed_deflection,
    simulate_detailed_edge,
    simulate_detailed_ridges,
)

# Define simulations as a dictionary with names as keys to function objects
# alphabectic order here
simulations = {
    "branching": simulate_detailed_branching,
    "curve": simulate_detailed_curve,
    "deflection": simulate_detailed_deflection,
    "edge": simulate_detailed_edge,
    "ridges": simulate_detailed_ridges,
}

########################
### x_train & x_test ###
########################

# Load training inputs (once for all simulations)
x_train = torch.load("data/sim_data/x_train_lines_discretised_0to1.pt", weights_only = False).float()

# Generate x_test (long) once for all simulations
_, x_test = make_grid(N_SIDE)
# x_test is long format (N_SIDE ** 2, 2)

#################################
### LOOP 1 - over SIMULATIONS ###
#################################

# Make y_train_dict: Iterate over all simulation functions
for sim_name, sim_func in simulations.items():

    ########################
    ### y_train & y_test ###
    ########################

    # Generate training observations
    # NOTE: sim_func() needs to be on CPU, so we move x_train to CPU
    y_train = sim_func(x_train.cpu()).to(device)
    y_test = sim_func(x_test.cpu()).to(device)
    
    x_test = x_test.to(device)
    x_train = x_train.to(device)

    # Print details
    print(f"=== {sim_name.upper()} ===")
    print(f"Training inputs shape: {x_train.shape}")
    print(f"Training observations shape: {y_train.shape}")
    print(f"Training inputs dtype: {x_train.dtype}")
    print(f"Training inputs device: {y_train.device}")
    print(f"Training observations device: {y_train.device}")
    print()

    # Print details
    print(f"=== {sim_name.upper()} ===")
    print(f"Test inputs shape: {x_test.shape}")
    print(f"Test observations shape: {y_test.shape}")
    print(f"Test inputs dtype: {x_test.dtype}")
    print(f"Test inputs device: {x_test.device}")
    print(f"Test observations device: {y_test.device}")
    print()

    # NOTE: This is different to the real data experiments
    # calculate the mean magnitude of the test data as we use this to scale the noise
    sim_mean_magnitude_for_noise = torch.norm(y_test, dim = -1).mean().to(device)
    sim_noise = STD_GAUSSIAN_NOISE * sim_mean_magnitude_for_noise

    # Store metrics for the simulation (used for *metrics_summary* report and *metrics_per_run*)
    simulation_results = [] 

    ##################################
    ### LOOP 2 - over training run ###
    ##################################

    # NOTE: GPs don't train on batches, use full data

    for run in range(NUM_RUNS):

        print(f"\n--- Training Run {run + 1}/{NUM_RUNS} ---")

        # Additive noise model: independent Gaussian noise
        # For every run we have a FIXED NOISY TARGET. Draw from standard normal with appropriate std
        y_train_noisy = y_train + (torch.randn(y_train.shape, device = device) * sim_noise)

        # Initialise the likelihood for the GP model (estimates noise)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        # Intialise fresh GP model with flat x_train and y_train_noisy (block-flat)
        model = dfNGP(
            x_train.T.reshape(-1),
            y_train_noisy.T.reshape(-1), 
            likelihood,
            ).to(device)
        
        model.covar
        
        optimizer = torch.optim.AdamW([
            {"params": model.mean_module.parameters(), "weight_decay": WEIGHT_DECAY, "lr": (10 * MODEL_LEARNING_RATE)},
            # {"params": list(model.covar_module.parameters()) + list(model.likelihood.parameters()), "weight_decay": # WEIGHT_DECAY, "lr": MODEL_LEARNING_RATE},
            ])
                
        # Use ExactMarginalLogLikelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        model.train()
        likelihood.train()
        
        # _________________
        # BEFORE EPOCH LOOP

        # Export the convergence just for first run only
        if run == 0:
            # initialise tensors to store losses over epochs (for convergence plot)
            train_losses_NLML_over_epochs = torch.zeros(MAX_NUM_EPOCHS) # objective
            train_losses_RMSE_over_epochs = torch.zeros(MAX_NUM_EPOCHS) # by-product
            # monitor performance transfer to test (only RMSE easy to calc without covar)
            test_losses_RMSE_over_epochs = torch.zeros(MAX_NUM_EPOCHS)

            sigma_n_over_epochs = torch.zeros(MAX_NUM_EPOCHS)
            sigma_f_over_epochs = torch.zeros(MAX_NUM_EPOCHS)
            l1_over_epochs = torch.zeros(MAX_NUM_EPOCHS)
            l2_over_epochs = torch.zeros(MAX_NUM_EPOCHS)

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
            # model outputs a multivariate normal distribution
            train_pred_dist = model(x_train.T.reshape(-1).to(device))
            # Train on noisy or true targets?
            loss = - mll(train_pred_dist, y_train_noisy.T.reshape(-1).to(device))  # negative marginal log likelihood
            loss.backward()
            optimizer.step()

            # For Run 1 we save a bunch of metrics and update, while for the rest we only update
            if run == 0:

                model.eval()
                likelihood.eval()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", gpytorch.utils.warnings.GPInputWarning)
                    train_pred_dist = model(x_train.T.reshape(-1).to(device))
                test_pred_dist = model(x_test.T.reshape(-1).to(device))

                # Compute RMSE for training and test predictions (given true data, not noisy)
                train_RMSE = torch.sqrt(gpytorch.metrics.mean_squared_error(train_pred_dist, y_train.T.reshape(-1).to(device)))
                test_RMSE = torch.sqrt(gpytorch.metrics.mean_squared_error(test_pred_dist, y_test.T.reshape(-1).to(device)))

                test_mean_module_RMSE = (model.mean_module(x_test.to(device)).detach() - y_test.to(device)).norm(dim = 1).mean()

                # Save losses for convergence plot
                train_losses_NLML_over_epochs[epoch] = loss.item()
                train_losses_RMSE_over_epochs[epoch] = train_RMSE.item()
                test_losses_RMSE_over_epochs[epoch] = test_RMSE.item()

                # Save evolution of hypers for convergence plot
                sigma_n_over_epochs[epoch] = model.likelihood.noise.item()
                sigma_f_over_epochs[epoch] = model.covar_module.outputscale.item()
                l1_over_epochs[epoch] = model.covar_module.lengthscale[0].item()
                l2_over_epochs[epoch] = model.covar_module.lengthscale[1].item()

                # Print a bit more information for the first run
                if epoch % 20 == 0:
                    print(f"{sim_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training Loss (NLML): {loss:.4f}, (RMSE): {train_RMSE:.4f}")
                    print(f"Test RMSE: {test_RMSE:.4f}, Test Mean Module RMSE: {test_mean_module_RMSE:.4f}")

                # delete after printing and saving
                # NOTE: keep loss for early stopping check
                del train_pred_dist, test_pred_dist, train_RMSE, test_RMSE
                
                # Free up memory every 20 epochs
                if epoch % 20 == 0:
                    gc.collect() and torch.cuda.empty_cache()
            
            # For all runs after the first we run a minimal version using only lml_train
            else:
                if epoch % 20 == 0:
                    # After run 1 we only print lml, nothing else
                    print(f"{sim_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training Loss (NLML): {loss:.4f}")
                
            # EVERY EPOCH: Early stopping check
            if loss < best_loss:
                best_loss = loss
                # reset counter if loss improves
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                # exit epoch loop
                break

        ##############################
        ### END LOOP 3 over EPOCHS ###
        ##############################

        # for every run...
        #######################################################
        ### EVALUATE after all training for RUN is finished ###
        #######################################################

        model.eval()
        likelihood.eval()

        # Need gradients for autograd divergence: We clone and detach
        x_test_grad = x_test.to(device).clone().requires_grad_(True)
        x_train_grad = x_train.to(device).clone().requires_grad_(True)

        # Underlying (latent) distribution and predictive distribution
        dist_test = model(x_test_grad.T.reshape(-1))
        pred_dist_test = likelihood(dist_test)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", gpytorch.utils.warnings.GPInputWarning)
            dist_train = model(x_train_grad.T.reshape(-1))
            pred_dist_train = likelihood(dist_train)
        
        # Compute divergence field (from latent distribution)
        test_div_field = compute_divergence_field(dist_test.mean.reshape(2, -1).T, x_test_grad)
        train_div_field = compute_divergence_field(dist_train.mean.reshape(2, -1).T, x_train_grad)

        # Only save mean_pred, covar_pred and divergence fields for the first run
        if run == 0:

            # (1) Save predictions from first run so we can visualise them later
            torch.save(pred_dist_test.mean.reshape(2, -1).T, f"{MODEL_SIM_RESULTS_DIR}/{sim_name}_{model_name}_test_mean_predictions.pt")
            torch.save(pred_dist_test.covariance_matrix, f"{MODEL_SIM_RESULTS_DIR}/{sim_name}_{model_name}_test_covar_predictions.pt")

            # (2) Save divergence field
            torch.save(test_div_field, f"{MODEL_SIM_RESULTS_DIR}/{sim_name}_{model_name}_test_prediction_divergence_field.pt")

            # (3) Since all epoch training is finished, we can save the losses over epochs
            df_losses = pd.DataFrame({
                'Epoch': list(range(train_losses_NLML_over_epochs.shape[0])), # pythonic indexing
                'Train NLML': train_losses_NLML_over_epochs.tolist(),
                'Train RMSE': train_losses_RMSE_over_epochs.tolist(),
                'Test RMSE': test_losses_RMSE_over_epochs.tolist(),
                'Sigma_n': sigma_n_over_epochs.tolist(),
                'Sigma_f': sigma_f_over_epochs.tolist(),
                'l1': l1_over_epochs.tolist(),
                'l2': l2_over_epochs.tolist()
                })
            
            df_losses.to_csv(f"{MODEL_SIM_RESULTS_DIR}/{sim_name}_{model_name}_losses_over_epochs.csv", index = False, float_format = "%.5f") # reduce to 5 decimals for readability

        # Compute TRAIN metrics (convert tensors to float) for every run's tuned model
        train_RMSE = torch.sqrt(gpytorch.metrics.mean_squared_error(
            pred_dist_train, y_train.T.reshape(-1).to(device))).item()
        train_MAE = gpytorch.metrics.mean_absolute_error(
            pred_dist_train, y_train.T.reshape(-1).to(device)).item()
        train_NLPD = gpytorch.metrics.negative_log_predictive_density(
            pred_dist_train, y_train.T.reshape(-1).to(device)).item()
        train_QCE = gpytorch.metrics.quantile_coverage_error(
            pred_dist_train, y_train.T.reshape(-1), quantile = 95).item()
        ## NOTE: It is important to use the absolute value of the divergence field, since positive and negative deviations are violations and shouldn't cancel each other out 
        train_div = train_div_field.abs().mean().item()

        # Compute TEST metrics (convert tensors to float) for every run's tuned model
        test_RMSE = torch.sqrt(gpytorch.metrics.mean_squared_error(
            pred_dist_test, y_test.T.reshape(-1).to(device))).item()
        test_MAE = gpytorch.metrics.mean_absolute_error(
            pred_dist_test, y_test.T.reshape(-1).to(device)).item()
        test_NLPD = gpytorch.metrics.negative_log_predictive_density(
            pred_dist_test, y_test.T.reshape(-1).to(device)).item()
        test_QCE = gpytorch.metrics.quantile_coverage_error(
            pred_dist_test, y_test.T.reshape(-1), quantile = 95).item()
        ## NOTE: It is important to use the absolute value of the divergence field, since positive and negative deviations are violations and shouldn't cancel each other out 
        test_div = test_div_field.abs().mean().item()

        simulation_results.append([
            run + 1,
            train_RMSE, train_MAE, train_NLPD, train_QCE, train_div,
            test_RMSE, test_MAE, test_NLPD, test_QCE, test_div
        ])

        # clean up
        del dist_train, dist_test, pred_dist_train, pred_dist_test, test_div_field, train_div_field
        gc.collect()
        torch.cuda.empty_cache()

    ############################
    ### END LOOP 2 over RUNS ###
    ############################

    # Convert results to a Pandas DataFrame
    results_per_run = pd.DataFrame(
        simulation_results, 
        columns = ["Run", 
                   "Train RMSE", "Train MAE", "Train NLPD", "Train QCE", "Train MAD",
                   "Test RMSE", "Test MAE", "Test NLPD", "Test QCE", "Test MAD"])

    # Compute mean and standard deviation for each metric
    mean_std_df = results_per_run.iloc[:, 1:].agg(["mean", "std"]) # Exclude "Run" column

    # Add sim_name and model_name as columns in the DataFrame _metrics_summary to be able to copy df
    mean_std_df["simulation name"] = sim_name
    mean_std_df["model name"] = model_name

    # Save "_metrics_per_run.csv" to CSV
    path_to_metrics_per_run = os.path.join(MODEL_SIM_RESULTS_DIR, f"{sim_name}_{model_name}_metrics_per_run.csv")
    results_per_run.to_csv(path_to_metrics_per_run, index = False, float_format = "%.5f") # reduce to 5 decimals
    print(f"\nResults per run saved to {path_to_metrics_per_run}")

    # Save "_metrics_summary.csv" to CSV
    path_to_metrics_summary = os.path.join(MODEL_SIM_RESULTS_DIR, f"{sim_name}_{model_name}_metrics_summary.csv")
    mean_std_df.to_csv(path_to_metrics_summary, float_format = "%.5f") # reduce to 5 decimals
    print(f"\nMean & Std saved to {path_to_metrics_summary}")

###############################
### END LOOP 1 over REGIONS ###
###############################

#############################
### WALL time & GPU model ###
#############################

end_time = time.time()
# compute elapsed time
elapsed_time = end_time - start_time 
# convert elapsed time to minutes
elapsed_time_minutes = elapsed_time / 60

# also end emission tracking. Will be saved as emissions.csv
if TRACK_EMISSIONS_BOOL:
    tracker.stop()

if device == "cuda":
    # get name of GPU model
    gpu_name = torch.cuda.get_device_name(0)
else:
    gpu_name = "N/A"

print(f"Elapsed wall time: {elapsed_time:.4f} seconds")

# Define full path for the file
wall_time_and_gpu_path = os.path.join(MODEL_SIM_RESULTS_DIR, model_name + "_run_" "wall_time.txt")

# Save to the correct folder with both seconds and minutes
with open(wall_time_and_gpu_path, "w") as f:
    f.write(f"Elapsed wall time: {elapsed_time:.4f} seconds\n")
    f.write(f"Elapsed wall time: {elapsed_time_minutes:.2f} minutes\n")
    f.write(f"Device used: {device}\n")
    f.write(f"GPU model: {gpu_name}\n")

print(f"Wall time saved to {wall_time_and_gpu_path}.")