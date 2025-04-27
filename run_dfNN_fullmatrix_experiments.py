from NN_models import dfNN_fullmatrix_for_vmap
from simulate import simulate_convergence, simulate_branching, simulate_ridge, simulate_merge, simulate_deflection
from metrics import compute_RMSE, compute_MAE
from utils import set_seed

# Global file for training configs
from configs import PATIENCE, MAX_NUM_EPOCHS, NUM_RUNS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, N_SIDE, DFNN_FULLMATRIX_RESULTS_DIR, DFNN_LEARNING_RATE

import torch
from torch.func import vmap, jacfwd
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

model_name = "dfNN_fullmatrix"

#########################
### x_train & y_train ###
#########################

# Import all simulation functions
from simulate import (
    simulate_convergence,
    simulate_branching,
    simulate_merge,
    simulate_deflection,
    simulate_ridge,
)

# Define simulations as a dictionary with names as keys to function objects
simulations = {
    "convergence": simulate_convergence,
    "branching": simulate_branching,
    "merge": simulate_merge,
    "deflection": simulate_deflection,
    "ridge": simulate_ridge,
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
MAX_NUM_EPOCHS = MAX_NUM_EPOCHS

# Number of training runs for mean and std of metrics
NUM_RUNS = NUM_RUNS
LEARNING_RATE = DFNN_LEARNING_RATE # trying 0.001 here instead
WEIGHT_DECAY = WEIGHT_DECAY

BATCH_SIZE = BATCH_SIZE

# Ensure the results folder exists
RESULTS_DIR = DFNN_FULLMATRIX_RESULTS_DIR
os.makedirs(RESULTS_DIR, exist_ok = True)

### LOOP OVER SIMULATIONS ###
for sim_name, sim_func in simulations.items():
    print(f"\nTraining for {sim_name.upper()}...")

    # Store metrics for the current simulation
    simulation_results = []

    # x_train is the same, select y_train
    y_train = y_train_dict[sim_name]
    # select the correct y_test (PREVIOUS ERROR)
    y_test = y_test_dict[sim_name]

    ### LOOP OVER RUNS ###
    for run in range(NUM_RUNS):
        print(f"\n--- Training Run {run + 1}/{NUM_RUNS} ---")

        # Convert to DataLoader for batching
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

        # Initialise fresh model
        # we seeded so this is reproducible
        dfNN_model = dfNN_fullmatrix_for_vmap().to(device)
        dfNN_model.train()

        # Define loss function (e.g., MSE for regression)
        criterion = torch.nn.MSELoss()

        # Define optimizer (e.g., AdamW)
        optimizer = optim.AdamW(dfNN_model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

        # Initialise tensors to store losses for this run
        epoch_train_losses = torch.zeros(MAX_NUM_EPOCHS)
        epoch_test_losses = torch.zeros(MAX_NUM_EPOCHS)

        # Early stopping variables
        best_loss = float('inf')
        epochs_no_improve = 0

        ### LOOP OVER EPOCHS ###
        print("\nStart Training")
        for epoch in range(MAX_NUM_EPOCHS):

            epoch_train_loss = 0.0  # Accumulate batch losses within epoch
            epoch_test_loss = 0.0

            for batch in dataloader:
                # assure model is in training mode 
                dfNN_model.train()

                x_batch, y_batch = batch
                # put on GPU
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                x_batch.requires_grad_()

                # Forward pass
                y_pred = vmap(dfNN_model)(x_batch)

                # Compute loss (RMSE for same units as data) - criterion(pred, target)
                loss = torch.sqrt(criterion(y_pred, y_batch))
                epoch_train_loss += loss.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ### END BATCH LOOP ###
            dfNN_model.eval()
            # Compute test loss for loss convergence plot
            y_test_pred = vmap(dfNN_model)(x_test.to(device))
            test_loss = torch.sqrt(criterion(y_test_pred, y_test.to(device))).item()
            
            # Compute average loss for the epoch (e.g. 7 batches/epoch)
            avg_train_loss = epoch_train_loss / len(dataloader)

            epoch_train_losses[epoch] = avg_train_loss
            epoch_test_losses[epoch] = test_loss

            print(f"{sim_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training Loss (RMSE): {avg_train_loss:.4f}")

            # debug
            # Early stopping check
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                epochs_no_improve = 0  # Reset counter
                best_model_state = dfNN_model.state_dict()  # Save best model
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
        
        ### END EPOCH LOOP ###
        # Load the best model before stopping for this "run"
        dfNN_model.load_state_dict(best_model_state)
        print(f"Run {run + 1}/{NUM_RUNS}, Training of {model_name} complete for {sim_name.upper()}. Restored best model.")

        ################
        ### EVALUATE ###
        ################

        # Evaluate the trained model after epochs are finished
        dfNN_model.eval()

        y_train_dfNN_predicted = vmap(dfNN_model)(x_train.to(device)).detach()
        y_test_dfNN_predicted = vmap(dfNN_model)(x_test.to(device)).detach()

        # Only save things for one run
        if run == 0:
            #(1) Save predictions from first run so we can visualise them later
            torch.save(y_test_dfNN_predicted, f"{RESULTS_DIR}/{sim_name}_{model_name}_test_predictions.pt")

            #(2) Save loss over epochs
            df_losses = pd.DataFrame({
                'Epoch': list(range(epoch_train_losses.shape[0])), # pythonic
                'Train Loss RMSE': epoch_train_losses.tolist(), 
                'Test Loss RMSE': epoch_test_losses.tolist()
                })
            
            df_losses.to_csv(f"{RESULTS_DIR}/{sim_name}_{model_name}_losses_over_epochs.csv", index = False)

        # Compute Divergence (convert tensor to float)
        dfNN_train_div = torch.diagonal(vmap(jacfwd(dfNN_model))(x_train.to(device)), dim1 = -2, dim2 = -1).detach().sum().item()
        dfNN_test_div = torch.diagonal(vmap(jacfwd(dfNN_model))(x_test.to(device)), dim1 = -2, dim2 = -1).detach().sum().item()

        # Compute metrics (convert tensors to float)
        dfNN_train_RMSE = compute_RMSE(y_train, y_train_dfNN_predicted.cpu()).item()
        dfNN_train_MAE = compute_MAE(y_train, y_train_dfNN_predicted.cpu()).item()

        dfNN_test_RMSE = compute_RMSE(y_test, y_test_dfNN_predicted.cpu()).item()
        dfNN_test_MAE = compute_MAE(y_test, y_test_dfNN_predicted.cpu()).item()

        # Store results in list
        simulation_results.append([
            run + 1, dfNN_train_RMSE, dfNN_train_MAE, dfNN_train_div,
            dfNN_test_RMSE, dfNN_test_MAE, dfNN_test_div
        ])

    ### FINISH LOOP OVER RUNS ###
    # Convert results to a Pandas DataFrame
    df = pd.DataFrame(
        simulation_results, 
        columns = ["Run", 
                   "Train RMSE", "Train MAE", "Train Divergence",
                   "Test RMSE", "Test MAE", "Test Divergence"])

    # Compute mean and standard deviation for each metric
    mean_std_df = df.iloc[:, 1:].agg(["mean", "std"])  # Exclude "Run" column

    # Save results to CSV
    results_file = os.path.join(RESULTS_DIR, f"{sim_name}_{model_name}_metrics_per_run.csv")
    df.to_csv(results_file, index = False)
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