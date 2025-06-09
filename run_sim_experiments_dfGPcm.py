# SIMULATED DATA EXPERIMENTS
# # RUN WITH python run_sim_experiments_dfGPcm.py
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

model_name = "dfGPcm"

# import configs to we can access the hypers with getattr
import configs
from configs import PATIENCE, MAX_NUM_EPOCHS, NUM_RUNS, WEIGHT_DECAY
# also import x_test grid size and std noise for training data
from configs import N_SIDE, STD_GAUSSIAN_NOISE
from configs import TRACK_EMISSIONS_BOOL

# Reiterating import for visibility
MAX_NUM_EPOCHS = MAX_NUM_EPOCHS
NUM_RUNS = NUM_RUNS
WEIGHT_DECAY = WEIGHT_DECAY
PATIENCE = PATIENCE

# assign model-specific variable
MODEL_LEARNING_RATE = getattr(configs, f"{model_name}_SIM_LEARNING_RATE")
MODEL_SIM_RESULTS_DIR = getattr(configs, f"{model_name}_SIM_RESULTS_DIR")
import os
os.makedirs(MODEL_SIM_RESULTS_DIR, exist_ok = True)

# imports for probabilistic models
if model_name in ["GP", "dfGP", "dfGPcm", "dfNGP"]:
    from GP_models import GP_predict
    from metrics import compute_NLL_sparse, compute_NLL_full
    from configs import L_RANGE, SIGMA_N_RANGE, GP_PATIENCE
    # overwrite with GP_PATIENCE
    PATIENCE = GP_PATIENCE

    if model_name in ["dfGP", "dfGPcm", "dfNGP"]:
        from configs import SIGMA_F_RANGE

# universals 
from metrics import compute_RMSE, compute_MAE, compute_divergence_field

# basics
import pandas as pd
import torch
import torch.nn as nn # NOTE: we also use this module for GP params
import torch.optim as optim
import gpytorch

# utilitarian
from utils import set_seed, make_grid
# reproducibility
set_seed(42)
import gc

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
    tracker = EmissionsTracker(project_name = "dfGPcm_simulation_experiments", output_dir = MODEL_SIM_RESULTS_DIR)
    tracker.start()

### SIMULATION ###
# Import all simulation functions
from simulate import (
    simulate_detailed_branching,
    # simulate_detailed_convergence,
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

        ### Initialise GP hyperparameters ###
        # 3 learnable HPs
        # NOTE: at every run this initialisation changes, introducing some randomness
        # HACK: we need to use nn.Parameter for trainable hypers to avoid leaf variable error

        # initialising (trainable) noise scalar from a uniform distribution over a predefined range
        sigma_n = nn.Parameter(torch.empty(1, device = device).uniform_( * SIGMA_N_RANGE)) # Trainable

        # initialising (trainable) output scalar from a uniform distribution over a predefined range
        sigma_f = nn.Parameter(torch.empty(1, device = device).uniform_( * SIGMA_F_RANGE))

        # initialising (trainable) lengthscales from a uniform distribution over a predefined range
        # each dimension has its own lengthscale
        l = nn.Parameter(torch.empty(2, device = device).uniform_( * L_RANGE))

        # AdamW as optimizer for some regularisation/weight decay
        optimizer = optim.AdamW([sigma_n, sigma_f, l], lr = MODEL_LEARNING_RATE, weight_decay = WEIGHT_DECAY)
        # NOTE: No need to initialise GP model like we initialise a NN model in torch
        
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

        # Additive noise model: independent Gaussian noise
        # For every run we have a FIXED NOISY TARGET. Draw from standard normal with appropriate std
        y_train_noisy = y_train + (torch.randn(y_train.shape, device = device) * sim_noise)

        mean_vector = y_train_noisy.mean(dim = 0)

        ############################
        ### LOOP 3 - over EPOCHS ###
        ############################

        print("\nStart Training")

        for epoch in range(MAX_NUM_EPOCHS):

            # For Run 1 we save a bunch of metrics and update, while for the rest we only update
            if run == 0:
                mean_pred_train, _, lml_train = GP_predict(
                        x_train,
                        y_train_noisy,
                        x_train, # predict training data
                        [sigma_n, sigma_f, l], # list of (initial) hypers
                        mean_func = mean_vector, # no mean aka "zero-mean function"
                        divergence_free_bool = True) # ensures we use a df kernel

                # Compute test loss for loss convergence plot
                mean_pred_test, _, _ = GP_predict(
                        x_train,
                        y_train_noisy,
                        x_test.to(device), # have predictions for training data again
                        # HACK: This is rather an eval, so we use detached hypers to avoid the computational tree
                        [sigma_n.detach().clone(), sigma_f.detach().clone(), l.detach().clone()], 
                        mean_func = mean_vector, # no mean aka "zero-mean function"
                        divergence_free_bool = True) # ensures we use a df kernel
                
                # UPDATE HYPERS (after test loss is computed to use same model)
                optimizer.zero_grad() # don't accumulate gradients
                # negative for NLML. loss is always on train
                loss = - lml_train
                loss.backward()
                optimizer.step()
                
                # NOTE: it is important to detach here 
                train_RMSE = compute_RMSE(y_train.detach(), mean_pred_train.detach())
                test_RMSE = compute_RMSE(y_test.detach(), mean_pred_test.detach())

                # Save losses for convergence plot
                train_losses_NLML_over_epochs[epoch] = - lml_train
                train_losses_RMSE_over_epochs[epoch] = train_RMSE
                # NOTE: lml is always just given training data. There is no TEST NLML
                test_losses_RMSE_over_epochs[epoch] = test_RMSE

                # Save evolution of hyprs for convergence plot
                sigma_n_over_epochs[epoch] = sigma_n[0]
                sigma_f_over_epochs[epoch] = sigma_f[0]
                l1_over_epochs[epoch] = l[0]
                l2_over_epochs[epoch] = l[1]

                print(f"{sim_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training Loss (NLML): {loss:.4f}, (RMSE): {train_RMSE:.4f}")

                # delete after printing and saving
                # NOTE: keep loss for early stopping check
                del mean_pred_train, mean_pred_test, lml_train, train_RMSE, test_RMSE
                
                # Free up memory every 20 epochs
                if epoch % 20 == 0:
                    gc.collect() and torch.cuda.empty_cache()
            
            # For all runs after the first we run a minimal version using only lml_train
            else:
                
                # NOTE: We can use x_train[0:2] since the predictions doesn;t matter and we only care about lml_train
                _, _, lml_train = GP_predict(
                        x_train,
                        y_train_noisy,
                        x_train[0:2], # predictions don't matter and we output lml_train already
                        [sigma_n, sigma_f, l], # list of (initial) hypers
                        mean_func = mean_vector, # no mean aka "zero-mean function"
                        divergence_free_bool = True) # ensures we use a df kernel
                
                # UPDATE HYPERS (after test loss is computed to use same model)
                optimizer.zero_grad() # don't accumulate gradients
                # negative for NLML
                loss = - lml_train
                loss.backward()
                optimizer.step()

                # After run 1 we only print lml, nothing else
                print(f"{sim_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training Loss (NLML): {loss:.4f}")

                # NOTE: keep loss for early stopping check, del lml_train
                del lml_train
                
                # Free up memory every 20 epochs
                if epoch % 20 == 0:
                    gc.collect() and torch.cuda.empty_cache()
                
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

        # Evaluate the trained model after all epochs are finished or early stopping was triggered
        # NOTE: Detach tuned hyperparameters from the computational graph
        best_sigma_n = sigma_n.detach().clone()
        best_sigma_f = sigma_f.detach().clone()
        best_l = l.detach().clone()

        # Need gradients for autograd divergence: We clone and detach
        x_test_grad = x_test.to(device).clone().requires_grad_(True)

        mean_pred_test, covar_pred_test, _ = GP_predict(
            x_train,
            y_train, # NOTE: use original y_train, not y_train_noisy
            x_test_grad,
            [best_sigma_n, best_sigma_f, best_l], # list of (initial) hypers
            mean_func = mean_vector, # no mean aka "zero-mean function"
            divergence_free_bool = True) # ensures we use a df kernel
        
        # Compute divergence field
        dfGPcm_test_div_field = compute_divergence_field(mean_pred_test, x_test_grad)

        # Only save mean_pred, covar_pred and divergence fields for the first run
        if run == 0:

            # (1) Save predictions from first run so we can visualise them later
            torch.save(mean_pred_test, f"{MODEL_SIM_RESULTS_DIR}/{sim_name}_{model_name}_test_mean_predictions.pt")
            torch.save(covar_pred_test, f"{MODEL_SIM_RESULTS_DIR}/{sim_name}_{model_name}_test_covar_predictions.pt")

            # (2) Save best hyperparameters
            # Stack tensors into a single tensor
            best_hypers_tensor = torch.cat([
                best_sigma_n.reshape(-1),  # Ensure 1D shape
                best_sigma_f.reshape(-1),
                best_l.reshape(-1),
            ])

            torch.save(best_hypers_tensor, f"{MODEL_SIM_RESULTS_DIR}/{sim_name}_{model_name}_best_hypers.pt")

            # (3) Since all epoch training is finished, we can save the losses over epochs
            df_losses = pd.DataFrame({
                'Epoch': list(range(train_losses_NLML_over_epochs.shape[0])), # pythonic indexing
                'Train Loss NLML': train_losses_NLML_over_epochs.tolist(),
                'Train Loss RMSE': train_losses_RMSE_over_epochs.tolist(),
                'Test Loss RMSE': test_losses_RMSE_over_epochs.tolist(),
                'Sigma_n': sigma_n_over_epochs.tolist(),
                'Sigma_f': sigma_f_over_epochs.tolist(),
                'l1': l1_over_epochs.tolist(),
                'l2': l2_over_epochs.tolist()
                })
            
            df_losses.to_csv(f"{MODEL_SIM_RESULTS_DIR}/{sim_name}_{model_name}_losses_over_epochs.csv", index = False, float_format = "%.5f") # reduce to 5 decimals for readability

            # (4) Save divergence field (computed above for all runs)
            torch.save(dfGPcm_test_div_field, f"{MODEL_SIM_RESULTS_DIR}/{sim_name}_{model_name}_test_prediction_divergence_field.pt")

        x_train_grad = x_train.to(device).clone().requires_grad_(True)

        mean_pred_train, covar_pred_train, _ = GP_predict(
                     x_train,
                     y_train, # NOTE: use original y_train, not y_train_noisy
                     x_train_grad,
                     [best_sigma_n, best_sigma_f, best_l], # list of (initial) hypers
                     mean_func = mean_vector, # no mean aka "zero-mean function"
                     divergence_free_bool = True) # ensures we use a df kernel
        
        dfGPcm_train_div_field = compute_divergence_field(mean_pred_train, x_train_grad)

        # Divergence: Convert field to metric: mean absolute divergence
        # NOTE: It is important to use the absolute value of the divergence field, since positive and negative deviations are violations and shouldn't cancel each other out 
        dfGPcm_train_div = dfGPcm_train_div_field.abs().mean().item()
        dfGPcm_test_div = dfGPcm_test_div_field.abs().mean().item()

        # Compute metrics (convert tensors to float) for every run's tuned model
        dfGPcm_train_RMSE = compute_RMSE(y_train, mean_pred_train).item()
        dfGPcm_train_MAE = compute_MAE(y_train, mean_pred_train).item()
        dfGPcm_train_sparse_NLL = compute_NLL_sparse(y_train, mean_pred_train, covar_pred_train).item()
        dfGPcm_train_full_NLL, dfGPcm_train_jitter = compute_NLL_full(y_train, mean_pred_train, covar_pred_train)
        # quantile coverage error
        pred_dist_train = gpytorch.distributions.MultivariateNormal(mean_pred_train.T.reshape(-1), covar_pred_train)
        dfGPcm_train_QCE = gpytorch.metrics.quantile_coverage_error(pred_dist_train, y_train.T.reshape(-1), quantile = 95).item()

        dfGPcm_test_RMSE = compute_RMSE(y_test, mean_pred_test).item()
        dfGPcm_test_MAE = compute_MAE(y_test, mean_pred_test).item()
        dfGPcm_test_sparse_NLL = compute_NLL_sparse(y_test, mean_pred_test, covar_pred_test).item()
        dfGPcm_test_full_NLL, dfGPcm_test_jitter = compute_NLL_full(y_test, mean_pred_test, covar_pred_test)
        # quantile coverage error
        pred_dist_test = gpytorch.distributions.MultivariateNormal(mean_pred_test.T.reshape(-1), covar_pred_test)
        dfGPcm_test_QCE = gpytorch.metrics.quantile_coverage_error(pred_dist_test, y_test.T.reshape(-1), quantile = 95).item()

        print(dfGPcm_train_jitter)
        print(dfGPcm_test_jitter)

        simulation_results.append([
            run + 1,
            dfGPcm_train_RMSE, dfGPcm_train_MAE, dfGPcm_train_sparse_NLL, dfGPcm_train_full_NLL.item(), dfGPcm_train_jitter.item(), dfGPcm_train_QCE, dfGPcm_train_div,
            dfGPcm_test_RMSE, dfGPcm_test_MAE, dfGPcm_test_sparse_NLL, dfGPcm_test_full_NLL.item(), dfGPcm_test_jitter.item(), dfGPcm_test_QCE, dfGPcm_test_div
        ])

        # clean up
        del mean_pred_train, mean_pred_test, covar_pred_train, covar_pred_test
        gc.collect()
        torch.cuda.empty_cache()

    ############################
    ### END LOOP 2 over RUNS ###
    ############################

    # Convert results to a Pandas DataFrame
    results_per_run = pd.DataFrame(
        simulation_results, 
        columns = ["Run", 
                   "Train RMSE", "Train MAE", "Train sparse NLL", "Train full NLL", "Train jitter", "Train QCE", "Train MAD",
                   "Test RMSE", "Test MAE", "Test sparse NLL", "Test full NLL", "Test jitter", "Test QCE", "Test MAD"])

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