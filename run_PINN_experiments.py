from archive.NN_models_HNN_dfNN_fullmatrix import PINN_backbone
# from simulate import simulate_convergence, simulate_branching, simulate_ridge, simulate_merge, simulate_deflection
from metrics import compute_RMSE, compute_MAE
from utils import set_seed

# Global file for training configs
from configs import PATIENCE, MAX_NUM_EPOCHS, NUM_RUNS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, N_SIDE, PINN_RESULTS_DIR, W_PINN_DIV_WEIGHT, PINN_LEARNING_RATE, STD_GAUSSIAN_NOISE

import torch
from torch.func import vmap
from torch.utils.data import DataLoader, TensorDataset
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

model_name = "PINN"

#########################
### x_train & y_train ###
#########################

# Import all simulation functions
from simulate import (
    simulate_detailed_edge,
    simulate_detailed_convergence,
    simulate_detailed_deflection,
    simulate_detailed_curve,
    simulate_detailed_ridges,
    simulate_detailed_branching,
)

# Define simulations as a dictionary with names as keys to function objects
simulations = {
    "edge": simulate_detailed_edge,
    "curve": simulate_detailed_curve,
    "deflection": simulate_detailed_deflection,
    "ridges": simulate_detailed_ridges,
    "branching": simulate_detailed_branching,
    "convergence": simulate_detailed_convergence,
}

# Load training inputs, not weights_only = True
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

#####################
### Training loop ###
#####################

# Early stopping parameters
PATIENCE = PATIENCE  # Stop after 50 epochs with no improvement

MAX_NUM_EPOCHS = MAX_NUM_EPOCHS # 2000

# Number of training runs for mean and std of metrics
NUM_RUNS = NUM_RUNS # 10

# higher lr for PINN
LEARNING_RATE = PINN_LEARNING_RATE
WEIGHT_DECAY = WEIGHT_DECAY

BATCH_SIZE = BATCH_SIZE

w = W_PINN_DIV_WEIGHT

# Ensure the results folder exists
RESULTS_DIR = PINN_RESULTS_DIR # Change this to "results" for full training

os.makedirs(RESULTS_DIR, exist_ok = True)

### LOOP OVER SIMULATIONS ###
for sim_name, sim_func in simulations.items():
    print(f"\nTraining for {sim_name.upper()}...")

    # Store metrics for the current simulation
    simulation_results = []

    # x_train, x_test stays the same but select y_train
    y_train = y_train_dict[sim_name]
    # select the correct y_test (PREVIOUS ERROR)
    y_test = y_test_dict[sim_name]

    ### LOOP OVER RUNS ###
    for run in range(NUM_RUNS):
        print(f"\n--- Training Run {run + 1}/{NUM_RUNS} ---")

        # Add Noise before data loader is defined
        y_train_noisy = y_train + (torch.randn(y_train.shape, device = device) * STD_GAUSSIAN_NOISE)

        # Convert to DataLoader for batching
        dataset = TensorDataset(x_train, y_train_noisy)
        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

        # Initialise fresh model
        # we seeded so this is reproducible
        PINN_model = PINN_backbone().to(device)
        PINN_model.train()

        # Define loss function (e.g., MSE for regression)
        criterion = torch.nn.MSELoss()

        # Define optimizer (e.g., AdamW)
        optimizer = optim.AdamW(PINN_model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

        # Initialise tensor to store losses
        epoch_train_losses = torch.zeros(MAX_NUM_EPOCHS)
        epoch_train_rmse_losses = torch.zeros(MAX_NUM_EPOCHS)
        epoch_test_losses = torch.zeros(MAX_NUM_EPOCHS)
        epoch_test_rmse_losses = torch.zeros(MAX_NUM_EPOCHS)

        # Early stopping variables
        best_loss = float('inf')
        epochs_no_improve = 0

        ### LOOP OVER EPOCHS ###
        print("\nStart Training")
        for epoch in range(MAX_NUM_EPOCHS):

            epoch_train_loss = 0.0  # Accumulate batch losses
            epoch_train_rmse_loss = 0.0

            for batch in dataloader:
                PINN_model.train()

                x_batch, y_batch = batch
                # put on GPU
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # inplace
                x_batch.requires_grad_()

                # Forward pass
                y_pred = vmap(PINN_model)(x_batch)

                u_indicator_batch, v_indicator_batch = torch.zeros_like(y_pred), torch.zeros_like(y_pred)
                u_indicator_batch[:, 0] = 1.0 # output column u selected
                v_indicator_batch[:, 1] = 1.0 # output column v selected

                # square(sum: f1/x1 + f2/x2)
                batch_divergence_loss = (torch.autograd.grad(
                    outputs = y_pred,
                    inputs = x_batch, # grad is on
                    grad_outputs = u_indicator_batch,
                    create_graph = True
                )[0][:, 0] + torch.autograd.grad(
                    outputs = y_pred,
                    inputs = x_batch,
                    grad_outputs = v_indicator_batch,
                    create_graph = True
                )[0][:, 1]).abs().mean().item() # v with respect to y
                # HERE: abs() to account for negative divergence and mean() to get avg!

                # Compute loss (RMSE for same units as data) + divergence loss
                loss = (1 - w) * torch.sqrt(criterion(y_pred, y_batch)) + w * batch_divergence_loss
                epoch_train_loss += loss.item()
                epoch_train_rmse_loss += torch.sqrt(criterion(y_pred, y_batch)).item()

                # Backpropagation
                # AFTER
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ### END BATCH LOOP ###
            PINN_model.eval()

            x_test_grad = x_test.to(device).requires_grad_()

            # Test loss outside of batch loop - once per epoch
            y_test_pred = vmap(PINN_model)(x_test_grad)
            # compute just once
            epoch_test_rmse_loss = torch.sqrt(criterion(y_test_pred, y_test.to(device))).item()

            # test divergence loss
            u_indicator_test, v_indicator_test = torch.zeros_like(y_test_pred), torch.zeros_like(y_test_pred)
            u_indicator_test[:, 0] = 1.0 # output column u selected
            v_indicator_test[:, 1] = 1.0 # output column v selected

            epoch_div_loss = (torch.autograd.grad(
                    outputs = y_test_pred,
                    inputs = x_test_grad, # grad is on
                    grad_outputs = u_indicator_test,
                    create_graph = True
                )[0][:, 0] + torch.autograd.grad(
                    outputs = y_test_pred,
                    inputs = x_test_grad,
                    grad_outputs = v_indicator_test,
                    create_graph = True
                )[0][:, 1]).abs().mean().item()

            # Put together
            epoch_test_loss = (1 - w) * torch.sqrt(criterion(y_test_pred, y_test.to(device))) + w * epoch_div_loss
 
            # Compute average loss for the epoch and store
            epoch_train_losses[epoch] = epoch_train_loss / len(dataloader)
            epoch_train_rmse_losses[epoch] = epoch_train_rmse_loss / len(dataloader)
            epoch_test_losses[epoch] = epoch_test_loss # old calc once per epoch over all
            epoch_test_rmse_losses[epoch] = epoch_test_rmse_loss

            print(f"{sim_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training Loss (RMSE): {epoch_train_rmse_losses[epoch]:.4f}, Training Loss (combined): {epoch_train_losses[epoch]:.4f}")

            # Early stopping check
            if epoch_train_losses[epoch] < best_loss:
                best_loss = epoch_train_losses[epoch]
                epochs_no_improve = 0  # Reset counter
                best_model_state = PINN_model.state_dict()  # Save best model
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        ### END EPOCH LOOP ###
        # Load the best model before stopping
        PINN_model.load_state_dict(best_model_state)
        print(f"Run {run + 1}/{NUM_RUNS}, Training of {model_name} complete for {sim_name.upper()}. Restored best model.")

        ################
        ### EVALUATE ###
        ################

        # Evaluate the trained model for each run
        PINN_model.eval()

        # We need to set requires_grad_() for the autograd divergence
        x_train_grad = x_train.to(device).requires_grad_()
        x_test_grad = x_test.to(device).requires_grad_()

        y_train_PINN_predicted = vmap(PINN_model)(x_train_grad)
        y_test_PINN_predicted = vmap(PINN_model)(x_test_grad)

        # Only save things for one run
        if run == 0:
            #(1) Save predictions from first run so we can visualise them later
            torch.save(y_test_PINN_predicted, f"{RESULTS_DIR}/{sim_name}_{model_name}_test_predictions.pt")

            #(2) Save losses over epochs
            df_losses = pd.DataFrame({
                'Epoch': list(range(epoch_train_losses.shape[0])), # pythonic
                'Train Loss': epoch_train_losses.tolist(),
                'Train Loss RMSE': epoch_train_rmse_losses.tolist(), 
                'Test Loss': epoch_test_losses.tolist(),
                'Test Loss RMSE': epoch_test_rmse_losses.tolist()
                })
            
            df_losses.to_csv(f"{RESULTS_DIR}/{sim_name}_{model_name}_losses_over_epochs.csv", index = False, float_format = "%.5f")

            #(3) Save (test) divergence field
            u_indicator_test, v_indicator_test = torch.zeros_like(y_test_PINN_predicted), torch.zeros_like(y_test_PINN_predicted)
            u_indicator_test[:, 0] = 1.0 # output column u selected
            v_indicator_test[:, 1] = 1.0 # output column v selected

            # divergence field (positive and negative divergences)
            PINN_test_div_field = (torch.autograd.grad(
                outputs = y_test_PINN_predicted,
                inputs = x_test_grad,
                grad_outputs = u_indicator_test,
                create_graph = True
            )[0][:, 0] + torch.autograd.grad(
                outputs = y_test_PINN_predicted,
                inputs = x_test_grad,
                grad_outputs = v_indicator_test,
                create_graph = True
            )[0][:, 1])

            # Save as test predition divergence field
            torch.save(PINN_test_div_field, f"{RESULTS_DIR}/{sim_name}_{model_name}_test_prediction_divergence_field.pt")

        # Compute Divergence (convert tensor to float)
        # Autograd divergence train
        u_indicator_train, v_indicator_train = torch.zeros_like(y_train_PINN_predicted), torch.zeros_like(y_train_PINN_predicted)
        u_indicator_train[:, 0] = 1.0 # output column u selected
        v_indicator_train[:, 1] = 1.0 # output column v selected

        PINN_train_div = (torch.autograd.grad(
            outputs = y_train_PINN_predicted,
            inputs = x_train_grad,
            grad_outputs = u_indicator_train,
            create_graph = True
        )[0][:, 0] + torch.autograd.grad(
            outputs = y_train_PINN_predicted,
            inputs = x_train_grad,
            grad_outputs = v_indicator_train,
            create_graph = True
        )[0][:, 1]).abs().mean().item() # v with respect to y

        # autograd div test
        u_indicator_test, v_indicator_test = torch.zeros_like(y_test_PINN_predicted), torch.zeros_like(y_test_PINN_predicted)
        u_indicator_test[:, 0] = 1.0 # output column u selected
        v_indicator_test[:, 1] = 1.0 # output column v selected

        PINN_test_div = (torch.autograd.grad(
            outputs = y_test_PINN_predicted,
            inputs = x_test_grad,
            grad_outputs = u_indicator_test,
            create_graph = True
        )[0][:, 0] + torch.autograd.grad(
            outputs = y_test_PINN_predicted,
            inputs = x_test_grad,
            grad_outputs = v_indicator_test,
            create_graph = True
        )[0][:, 1]).abs().mean().item() # v with respect to y

        # Compute metrics (convert tensors to float)
        dfNN_train_RMSE = compute_RMSE(y_train, y_train_PINN_predicted.cpu()).item()
        dfNN_train_MAE = compute_MAE(y_train, y_train_PINN_predicted.cpu()).item()

        dfNN_test_RMSE = compute_RMSE(y_test, y_test_PINN_predicted.cpu()).item()
        dfNN_test_MAE = compute_MAE(y_test, y_test_PINN_predicted.cpu()).item()

        # Store results in list
        simulation_results.append([
            run + 1, dfNN_train_RMSE, dfNN_train_MAE, PINN_train_div,
            dfNN_test_RMSE, dfNN_test_MAE, PINN_test_div
        ])

    ### FINISH LOOP OVER RUNS ###
    # Convert results to a Pandas DataFrame
    df = pd.DataFrame(
        simulation_results, 
        columns = ["Run", 
                   "Train RMSE", "Train MAE", "Train MAD",
                   "Test RMSE", "Test MAE", "Test MAD"])

    # Compute mean and standard deviation for each metric
    mean_std_df = df.iloc[:, 1:].agg(["mean", "std"])  # Exclude "Run" column

    # Add sim_name and model_name as columns in the DataFrame _metrics_summary
    mean_std_df["sim name"] = sim_name
    mean_std_df["model name"] = model_name

    # Save results to CSV
    results_file = os.path.join(RESULTS_DIR, f"{sim_name}_{model_name}_metrics_per_run.csv")
    df.to_csv(results_file, index = False, float_format = "%.5f")
    print(f"\nResults saved to {results_file}")

    # Save mean and standard deviation to CSV
    mean_std_file = os.path.join(RESULTS_DIR, f"{sim_name}_{model_name}_metrics_summary.csv")
    mean_std_df.to_csv(mean_std_file, float_format = "%.5f") # reduce to 5 decimals
    print(f"\nMean & Std saved to {mean_std_file}")

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