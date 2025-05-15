from NN_models import dfNN
from metrics import compute_RMSE, compute_MAE
from utils import set_seed

# Global file for training configs
from configs import PATIENCE, MAX_NUM_EPOCHS, NUM_RUNS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, dfNN_REAL_RESULTS_DIR

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os
import pandas as pd

# Set seed for reproducibility
set_seed(42)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

device = 'cpu'

model_name = "dfNN"

#########################
### x_train & y_train ###
#########################

### TIMING ###
import time
start_time = time.time()  # Start timing after imports

#########################
### Loop over regions ###
#########################

# What used to be region_name is now region name
# For region_name in ["regiona", "regionb", "regionc"]:
for region_name in ["regionc"]:

    print(f"\nTraining for {region_name.upper()}...")

    # Store metrics for the current simulation
    region_results = []

    #########################
    ### x_train & y_train ###
    #########################

    path_to_training_tensor = "data/real_data/" + region_name + "_train_tensor.pt"
    path_to_test_tensor = "data/real_data/" + region_name + "_test_tensor.pt"

    train = torch.load(path_to_training_tensor, weights_only = False).T # we need to transpose the tensor to have observations in the first dimension
    test = torch.load(path_to_test_tensor, weights_only = False).T

    # The train and test tensors have the following columns:
    # [:, 0] = x
    # [:, 1] = y
    # [:, 2] = surface elevation (s)
    # [:, 3] = ice flux in x direction (u)
    # [:, 4] = ice flux in y direction (v)
    # [:, 5] = ice flux error in x direction (u_err)
    # [:, 6] = ice flux error in y direction (v_err)

    x_train = train[:, [0, 1]].to(device)
    y_train = train[:, [3, 4]].to(device)

    x_test = test[:, [0, 1]].to(device)
    y_test = test[:, [3, 4]].to(device)

    train_noise_diag = torch.concat((train[:, 5], train[:, 5]), dim = 0).to(device)
    train_noise_diagmatrix = torch.eye(len(train_noise_diag)).to(device) * train_noise_diag

    # Print details
    print(f"=== {region_name.upper()} ===")
    print(f"Training inputs shape: {x_train.shape}")
    print(f"Training observations shape: {y_train.shape}")
    print(f"Training inputs dtype: {x_train.dtype}")
    print()

    # Print details
    print(f"=== {region_name.upper()} ===")
    print(f"Test inputs shape: {x_test.shape}")
    print(f"Test observations shape: {y_test.shape}")
    print(f"Test inputs dtype: {x_test.dtype}")
    print()

    #####################
    ### Training loop ###
    #####################

    # Early stopping parameters
    PATIENCE = PATIENCE
    MAX_NUM_EPOCHS = MAX_NUM_EPOCHS
    BATCH_SIZE = BATCH_SIZE

    # Number of training runs for mean and std of metrics
    NUM_RUNS = NUM_RUNS
    NUM_RUNS = 1
    LEARNING_RATE = LEARNING_RATE
    WEIGHT_DECAY = WEIGHT_DECAY

    # Ensure the results folder exists
    RESULTS_DIR = dfNN_REAL_RESULTS_DIR
    os.makedirs(RESULTS_DIR, exist_ok = True)
    ### LOOP OVER RUNS ###
    for run in range(NUM_RUNS):
        print(f"\n--- Training Run {run + 1}/{NUM_RUNS} ---")

        # Convert to DataLoader for batching
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

        # Initialise fresh model
        # we seeded so this is reproducible
        dfNN_model = dfNN().to(device)
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

        # We do not need to add extra noise to real data

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
                y_pred = dfNN_model(x_batch)

                # Compute loss (RMSE for same units as data) - criterion(pred, target)
                loss = torch.sqrt(criterion(y_pred, y_batch))
                epoch_train_loss += loss.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ### END LOOP OVER BATCHES ###

            dfNN_model.eval()
            # Compute test loss for loss convergence plot
            y_test_pred = dfNN_model(x_test.to(device).requires_grad_())
            test_loss = torch.sqrt(criterion(y_test_pred, y_test.to(device))).item()
            
            # Compute average loss for the epoch (e.g. 7 batches/epoch)
            avg_train_loss = epoch_train_loss / len(dataloader)

            epoch_train_losses[epoch] = avg_train_loss
            epoch_test_losses[epoch] = test_loss

            print(f"{region_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training Loss (RMSE): {avg_train_loss:.4f}")

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
### END LOOP OVER EPOCHS FOR THIS RUN ###
        # Load the best model before stopping for this "run"
        dfNN_model.load_state_dict(best_model_state)
        print(f"Run {run + 1}/{NUM_RUNS}, Training of {model_name} complete for {region_name.upper()}. Restored best model.")

        ################
        ### EVALUATE ###
        ################

        # Evaluate the trained model after epochs are finished
        dfNN_model.eval()

        # grad for autograd
        x_train_grad = x_train.to(device).requires_grad_()
        x_test_grad = x_test.to(device).requires_grad_()

        y_train_dfNN_predicted = dfNN_model(x_train_grad)
        y_test_dfNN_predicted = dfNN_model(x_test_grad)

        # Only save things for one run
        if run == 0:
            #(1) Save predictions from first run so we can visualise them later
            torch.save(y_test_dfNN_predicted, f"{RESULTS_DIR}/{region_name}_{model_name}_test_predictions.pt")

            #(2) Save loss over epochs
            df_losses = pd.DataFrame({
                'Epoch': list(range(epoch_train_losses.shape[0])), # pythonic
                'Train Loss RMSE': epoch_train_losses.tolist(), 
                'Test Loss RMSE': epoch_test_losses.tolist()
                })
            
            df_losses.to_csv(f"{RESULTS_DIR}/{region_name}_{model_name}_losses_over_epochs.csv", index = False, float_format = "%.5f")

            # No pretraining done

            #(3) Save (test) divergence field
            u_indicator_test, v_indicator_test = torch.zeros_like(y_test_dfNN_predicted), torch.zeros_like(y_test_dfNN_predicted)
            u_indicator_test[:, 0] = 1.0 # output column u selected
            v_indicator_test[:, 1] = 1.0 # output column v selected

            # divergence field (positive and negative divergences)
            dfNN_test_div_field = (torch.autograd.grad(
                outputs = y_test_dfNN_predicted,
                inputs = x_test_grad,
                grad_outputs = u_indicator_test,
                create_graph = True
            )[0][:, 0] + torch.autograd.grad(
                outputs = y_test_dfNN_predicted,
                inputs = x_test_grad,
                grad_outputs = v_indicator_test,
                create_graph = True
            )[0][:, 1])

            # Save as test predition divergence field
            torch.save(dfNN_test_div_field, f"{RESULTS_DIR}/{region_name}_{model_name}_test_prediction_divergence_field.pt")

        # autograd divergence train
        u_indicator_train, v_indicator_train = torch.zeros_like(y_train_dfNN_predicted), torch.zeros_like(y_train_dfNN_predicted)
        u_indicator_train[:, 0] = 1.0 # output column u selected
        v_indicator_train[:, 1] = 1.0 # output column v selected

        dfNN_train_div = (torch.autograd.grad(
            outputs = y_train_dfNN_predicted,
            inputs = x_train_grad,
            grad_outputs = u_indicator_train,
            create_graph = True
        )[0][:, 0] + torch.autograd.grad(
            outputs = y_train_dfNN_predicted,
            inputs = x_train_grad,
            grad_outputs = v_indicator_train,
            create_graph = True
        )[0][:, 1]).abs().mean().item() # v with respect to y

        # Divergence
        # autograd divergence test
        u_indicator_test, v_indicator_test = torch.zeros_like(y_test_dfNN_predicted), torch.zeros_like(y_test_dfNN_predicted)
        u_indicator_test[:, 0] = 1.0 # output column u selected
        v_indicator_test[:, 1] = 1.0 # output column v selected

        dfNN_test_div = (torch.autograd.grad(
            outputs = y_test_dfNN_predicted,
            inputs = x_test_grad,
            grad_outputs = u_indicator_test,
            create_graph = True
        )[0][:, 0] + torch.autograd.grad(
            outputs = y_test_dfNN_predicted,
            inputs = x_test_grad,
            grad_outputs = v_indicator_test,
            create_graph = True
        )[0][:, 1]).abs().mean().item() # v with respect to y

        # Compute metrics (convert tensors to float)
        dfNN_train_RMSE = compute_RMSE(y_train, y_train_dfNN_predicted).item()
        dfNN_train_MAE = compute_MAE(y_train, y_train_dfNN_predicted).item()

        dfNN_test_RMSE = compute_RMSE(y_test, y_test_dfNN_predicted).item()
        dfNN_test_MAE = compute_MAE(y_test, y_test_dfNN_predicted).item()

        # Store results in list
        region_results.append([
            run + 1, dfNN_train_RMSE, dfNN_train_MAE, dfNN_train_div,
            dfNN_test_RMSE, dfNN_test_MAE, dfNN_test_div
        ])

    ### FINISH LOOP OVER RUNS ###

    # Convert results to a Pandas DataFrame
    df = pd.DataFrame(
        region_results, 
        columns = ["Run", 
                   "Train RMSE", "Train MAE", "Train MAD",
                   "Test RMSE", "Test MAE", "Test MAD"])

    # Compute mean and standard deviation for each metric
    mean_std_df = df.iloc[:, 1:].agg(["mean", "std"])  # Exclude "Run" column

    # Add region_name and model_name as columns in the DataFrame _metrics_summary
    mean_std_df["sim name"] = region_name
    mean_std_df["model name"] = model_name

    # Save results to CSV
    results_file = os.path.join(RESULTS_DIR, f"{region_name}_{model_name}_metrics_per_run.csv")
    df.to_csv(results_file, index = False, float_format = "%.5f")
    print(f"\nResults saved to {results_file}")

    # Save mean and standard deviation to CSV
    mean_std_file = os.path.join(RESULTS_DIR, f"{region_name}_{model_name}_metrics_summary.csv")
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