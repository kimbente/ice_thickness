from GP_models import GP_predict
# from simulate import simulate_convergence, simulate_branching, simulate_ridge, simulate_merge, simulate_deflection
from utils import set_seed
from metrics import compute_RMSE, compute_NLL, compute_MAE

# Global file for training configs
from configs import PATIENCE, GP_MAX_NUM_EPOCHS, NUM_RUNS, GP_LEARNING_RATE, WEIGHT_DECAY, N_SIDE, GP_RESULTS_DIR, L_RANGE, B_DIAGONAL_RANGE, B_OFFDIAGONAL_RANGE

import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd

### TIMING ###
import time
start_time = time.time()  # Start timing after imports

# Set seed for reproducibility
set_seed(42)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

model_name = "GP"

#########################
### x_train & y_train ###
#########################

# Import all simulation functions
from simulate import (
    simulate_detailed_convergence,
    simulate_detailed_deflection,
    simulate_detailed_curve,
    simulate_detailed_ridges,
    simulate_detailed_branching,
)

# Define simulations as a dictionary with names as keys to function objects
simulations = {
    "convergence_dtl": simulate_detailed_convergence,
    "deflection_dtl": simulate_detailed_deflection,
    "curve_dtl": simulate_detailed_curve,
    "ridges_dtl": simulate_detailed_ridges,
    "branching_dtl": simulate_detailed_branching,
}

# Load training inputs
x_train = torch.load("data/sim_data/x_train_lines_discretised_0to1.pt", weights_only = False).float()

# Storage dictionaries
y_train_dict = {}

# Make y_train_dict: Iterate over all simulation functions
for sim_name, sim_func in simulations.items():

    # Generate training observations
    y_train = sim_func(x_train)
    y_train_dict[sim_name] = y_train  # Store training outputs

    # Print details
    print(f"=== {sim_name.upper()} ===")
    print(f"Training inputs shape: {x_train.shape}")
    print(f"Training observations shape: {y_train.shape}")
    print(f"Training inputs dtype: {x_train.dtype}")
    print()

#######################
### x_test & y_test ###
#######################

print("=== Generating test data ===")

# Choose discretisation that is good for simulations and also for quiver plotting
N_SIDE = N_SIDE

side_array = torch.linspace(start = 0.0, end = 1.0, steps = N_SIDE)
XX, YY = torch.meshgrid(side_array, side_array, indexing = "xy")
x_test_grid = torch.cat([XX.unsqueeze(-1), YY.unsqueeze(-1)], dim = -1)
# long format
x_test = x_test_grid.reshape(-1, 2)

# Storage dictionaries
y_test_dict = {}

# Make y_test_dict: Iterate over all simulation functions
for sim_name, sim_func in simulations.items():

    # Generate test observations
    y_test = sim_func(x_test)
    y_test_dict[sim_name] = y_test  # Store test outputs

    # Print details
    print(f"=== {sim_name.upper()} ===")
    print(f"Test inputs shape: {x_test.shape}")
    print(f"Test observations shape: {y_test.shape}")
    print(f"Test inputs dtype: {x_test.dtype}")
    print()

    # visualise_v_quiver(y_test, x_test, title_string = name)

#####################
### Training loop ###
#####################

# Early stopping parameters
PATIENCE = PATIENCE
MAX_NUM_EPOCHS = GP_MAX_NUM_EPOCHS

# Number of training runs for mean and std of metrics
NUM_RUNS = NUM_RUNS
LEARNING_RATE = GP_LEARNING_RATE
WEIGHT_DECAY = WEIGHT_DECAY

# Pass in all the training data
# BATCH_SIZE = BATCH_SIZE

# Ensure the results folder exists
RESULTS_DIR = GP_RESULTS_DIR
os.makedirs(RESULTS_DIR, exist_ok = True)

### LOOP OVER SIMULATIONS ###
for sim_name, sim_func in simulations.items():
    print(f"\nTraining for {sim_name.upper()}...")

    # Store metrics for the current simulation
    simulation_results = []

    # x_train is the same, select y_train
    x_train = x_train.to(device)

    y_train = y_train_dict[sim_name].to(device)
    # select the correct y_test (PREVIOUS ERROR)
    y_test = y_test_dict[sim_name].to(device)

    ### LOOP OVER RUNS ###
    for run in range(NUM_RUNS):
        print(f"\n--- Training Run {run + 1}/{NUM_RUNS} ---")

        # Sample from uniform distributions to initialise hyperparameters
        sigma_n = torch.tensor([0.05], requires_grad = False).to(device) # no optimisation for noise, no sampling
        sigma_f = torch.tensor([1.0], requires_grad = False).to(device) # Fixed because we tune B (and sigma_f would just scale B)
        # Initialising l from a uniform distribution as nn.Param to avoid leaf variable error
        l = nn.Parameter(torch.empty(2, device = device).uniform_( * L_RANGE))

        # Trainable B matrix components
        B_diag = nn.Parameter(torch.empty(1, device = device).uniform_( * B_DIAGONAL_RANGE))  
        B_off_diag = nn.Parameter(torch.empty(1, device = device).uniform_( * B_OFFDIAGONAL_RANGE)) 

        # Construct B using a proper tensor operation (not `torch.tensor()`)
        # squeeze to make 2 x 2 
        B = torch.stack([
            torch.stack([B_diag, B_off_diag], dim = 0),
            torch.stack([B_off_diag, B_diag], dim = 0)
        ], dim = 0).squeeze()

        B = nn.Parameter(B)
        
        # We do not need to "initialse" the GP model
        # We don't need a criterion either

        # Define optimizer (e.g., AdamW)
        optimizer = optim.AdamW([l, B], lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

        # Initialise tensors to store losses over epochs (for convergence plot)
        epoch_train_NLML_losses = torch.zeros(MAX_NUM_EPOCHS)
        epoch_train_RMSE_losses = torch.zeros(MAX_NUM_EPOCHS)
        epoch_test_RMSE_losses = torch.zeros(MAX_NUM_EPOCHS)

        epoch_sigma_f = torch.zeros(MAX_NUM_EPOCHS)
        epoch_l1 = torch.zeros(MAX_NUM_EPOCHS)
        epoch_l2 = torch.zeros(MAX_NUM_EPOCHS)

        # Early stopping variables
        best_loss = float('inf')
        epochs_no_improve = 0

        ### LOOP OVER EPOCHS ###
        print("\nStart Training")
        for epoch in range(MAX_NUM_EPOCHS):

            # No batching - full epoch pass in one
            if run == 0:
                mean_pred_train, _, lml_train = GP_predict(
                        x_train,
                        y_train,
                        x_train, # have predictions for training data again
                        [sigma_n, sigma_f, l, B], # initial hyperparameters
                        # no mean
                        divergence_free_bool = False)
                
                loss = - lml_train
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute test loss for loss convergence plot
                mean_pred_test, _, _ = GP_predict(
                        x_train,
                        y_train,
                        x_test.to(device), # have predictions for training data again
                        [sigma_n, sigma_f, l, B], # initial hyperparameters
                        # no mean
                        divergence_free_bool = False)
                
                train_RMSE = compute_RMSE(y_train, mean_pred_train)
                test_RMSE = compute_RMSE(y_test, mean_pred_test)

                epoch_train_NLML_losses[epoch] = - lml_train
                epoch_train_RMSE_losses[epoch] = train_RMSE
                # epoch_test_NLML_losses[epoch] =  # train NLML
                epoch_test_RMSE_losses[epoch] = test_RMSE

                epoch_sigma_f[epoch] = sigma_f
                epoch_l1[epoch] = l[0]
                epoch_l2[epoch] = l[1]

                print(f"{sim_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training Loss (NLML): {loss:.4f}, (RMSE): {train_RMSE:.4f}")
            
            else:
                # Save compute after run 1
                _, _, lml_train = GP_predict(
                        x_train,
                        y_train,
                        x_train[0:2], # have predictions for training data again
                        [sigma_n, sigma_f, l, B], # initial hyperparameters
                        # no mean
                        divergence_free_bool = False)
                
                loss = - lml_train
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"{sim_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training Loss (NLML): {loss:.4f}")

            # Early stopping check
            if loss < best_loss:
                best_loss = loss
                epochs_no_improve = 0  # Reset counter
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        ################
        ### EVALUATE ###
        ################

        # Now HPs should be tuned
        # Evaluate the trained model after all epochs are finished/early stopping

        # Need gradients for autograd divergence
        x_test_grad = x_test.to(device).requires_grad_(True)

        mean_pred_test, covar_pred_test, _ = GP_predict(
                     x_train,
                     y_train,
                     x_test_grad,
                     [sigma_n, sigma_f, l, B], # optimal hypers
                     # no mean
                     divergence_free_bool = False)

        # Only save things for one run
        if run == 0:
            #(1) Save predictions from first run so we can visualise them later
            torch.save(mean_pred_test, f"{RESULTS_DIR}/{sim_name}_{model_name}_test_mean_predictions.pt")
            torch.save(covar_pred_test, f"{RESULTS_DIR}/{sim_name}_{model_name}_test_covar_predictions.pt")

            #(2) Save best hyperparameters from run 1
            # Stack tensors into a single tensor
            best_hypers_tensor = torch.cat([
                sigma_n.reshape(-1),  # Ensure 1D shape
                sigma_f.reshape(-1),
                l.reshape(-1),
                B.reshape(-1)
            ])

            # Save the tensor
            torch.save(best_hypers_tensor, f"{RESULTS_DIR}/{sim_name}_{model_name}_best_hypers.pt")

            #(3) Save loss over epochs
            df_losses = pd.DataFrame({
                'Epoch': list(range(epoch_train_NLML_losses.shape[0])), # pythonic
                'Train Loss NLML': epoch_train_NLML_losses.tolist(),
                'Train Loss RMSE': epoch_train_RMSE_losses.tolist(),
                'Test Loss RMSE': epoch_test_RMSE_losses.tolist(),
                'Sigma_f': epoch_sigma_f.tolist(),
                'l1': epoch_l1.tolist(),
                'l2': epoch_l2.tolist()
                })
            
            df_losses.to_csv(f"{RESULTS_DIR}/{sim_name}_{model_name}_losses_over_epochs.csv", index = False, float_format = "%.5f") # reduce to 5 decimals

        x_train_grad = x_train.to(device).requires_grad_(True)

        mean_pred_train, covar_pred_train, _ = GP_predict(
                     x_train,
                     y_train,
                     x_train_grad,
                     [sigma_n, sigma_f, l, B], # optimal hypers
                     # no mean
                     divergence_free_bool = False)

        ### Divergence: Total absolute divergence (sum divergence at each point, after summing dims)
        # autograd div test
        u_indicator_test, v_indicator_test = torch.zeros_like(mean_pred_test), torch.zeros_like(mean_pred_test)
        u_indicator_test[:, 0] = 1.0 # output column u selected
        v_indicator_test[:, 1] = 1.0 # output column v selected

        GP_test_div = (torch.autograd.grad(
            outputs = mean_pred_test,
            inputs = x_test_grad,
            grad_outputs = u_indicator_test,
            create_graph = True
        )[0][:, 0] + torch.autograd.grad(
            outputs = mean_pred_test,
            inputs = x_test_grad,
            grad_outputs = v_indicator_test,
            create_graph = True
        )[0][:, 1]).abs().sum().item() # v with respect to y

        # autograd div train
        u_indicator_train, v_indicator_train = torch.zeros_like(mean_pred_train), torch.zeros_like(mean_pred_train)
        u_indicator_train[:, 0] = 1.0 # output column u selected
        v_indicator_train[:, 1] = 1.0 # output column v selected

        GP_train_div = (torch.autograd.grad(
            outputs = mean_pred_train,
            inputs = x_train_grad,
            grad_outputs = u_indicator_train,
            create_graph = True
        )[0][:, 0] + torch.autograd.grad(
            outputs = mean_pred_train,
            inputs = x_train_grad,
            grad_outputs = v_indicator_train,
            create_graph = True
        )[0][:, 1]).abs().sum().item() # v with respect to y

        # Compute metrics (convert tensors to float) for every run's tuned model
        GP_train_RMSE = compute_RMSE(y_train, mean_pred_train).item()
        GP_train_MAE = compute_MAE(y_train, mean_pred_train).item()
        GP_train_NLL = compute_NLL(y_train, mean_pred_train, covar_pred_train).item()

        GP_test_RMSE = compute_RMSE(y_test, mean_pred_test).item()
        GP_test_MAE = compute_MAE(y_test, mean_pred_test).item()
        # full NLL has caused instability issues due to the logdet
        # now we use sparse
        GP_test_NLL = compute_NLL(y_test, mean_pred_test, covar_pred_test).item()

        simulation_results.append([
            run + 1,
            GP_train_RMSE, GP_train_MAE, GP_train_NLL, GP_train_div,
            GP_test_RMSE, GP_test_MAE, GP_test_NLL, GP_test_div
        ])

    ### FINISH LOOP OVER RUNS ###
    # Convert results to a Pandas DataFrame
    df = pd.DataFrame(
        simulation_results, 
        columns = ["Run", 
                   "Train RMSE", "Train MAE", "Train NLL", "Train Divergence",
                   "Test RMSE", "Test MAE", "Test NLL", "Test Divergence"])

    # Compute mean and standard deviation for each metric
    mean_std_df = df.iloc[:, 1:].agg(["mean", "std"])  # Exclude "Run" column

    # Add sim_name and model_name as columns in the DataFrame _metrics_summary
    mean_std_df["sim name"] = sim_name
    mean_std_df["model name"] = model_name

    # Save results to CSV
    results_file = os.path.join(RESULTS_DIR, f"{sim_name}_{model_name}_metrics_per_run.csv")
    df.to_csv(results_file, index = False, float_format = "%.5f") # reduce to 5 decimals
    print(f"\nResults saved to {results_file}")

    # Save mean and standard deviation to CSV
    mean_std_file = os.path.join(RESULTS_DIR, f"{sim_name}_{model_name}_metrics_summary.csv")
    mean_std_df.to_csv(mean_std_file, float_format = "%.5f") # reduce to 5 decimals
    print(f"\nMean & Std saved to {mean_std_file}")
    # Only train for one simulation for now

### End timing ###
end_time = time.time()  # End timing
elapsed_time = end_time - start_time  # Compute elapsed time
# Convert elapsed time to minutes
elapsed_time_minutes = elapsed_time / 60

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)  # Get GPU model
else:
    gpu_name = "N/A"

print(f"Elapsed wall time: {elapsed_time:.4f} seconds")

# Define full path for the file
wall_time_path = os.path.join(RESULTS_DIR, model_name + "_run_" "wall_time.txt")

# Save to the correct folder with both seconds and minutes
with open(wall_time_path, "w") as f:
    f.write(f"Elapsed wall time: {elapsed_time:.4f} seconds\n")
    f.write(f"Elapsed wall time: {elapsed_time_minutes:.2f} minutes\n")
    f.write(f"Device used: {device}\n")
    f.write(f"GPU model: {gpu_name}\n")

print(f"Wall time saved to {wall_time_path}.")