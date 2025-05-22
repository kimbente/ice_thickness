# REAL DATA EXPERIMENTS
# RUN WITH python run_real_experiments_PINN.py
#               _                 _   _      
#              | |               | | (_)     
#    __ _ _ __ | |_ __ _ _ __ ___| |_ _  ___ 
#   / _` | '_ \| __/ _` | '__/ __| __| |/ __|
#  | (_| | | | | || (_| | | | (__| |_| | (__ 
#   \__,_|_| |_|\__\__,_|_|  \___|\__|_|\___|
# 
model_name = "PINN"

# import configs to we can access the hypers with getattr
import configs
from configs import PATIENCE, MAX_NUM_EPOCHS, NUM_RUNS, WEIGHT_DECAY

# Reiterating import for visibility
MAX_NUM_EPOCHS = MAX_NUM_EPOCHS
NUM_RUNS = NUM_RUNS
WEIGHT_DECAY = WEIGHT_DECAY
PATIENCE = PATIENCE

# TODO: Delete overwrite, run full
NUM_RUNS = 1

# assign model-specific variable
MODEL_LEARNING_RATE = getattr(configs, f"{model_name}_REAL_LEARNING_RATE")
MODEL_REAL_RESULTS_DIR = getattr(configs, f"{model_name}_REAL_RESULTS_DIR")
import os
os.makedirs(MODEL_REAL_RESULTS_DIR, exist_ok = True)

# for all models with NN components train on batches
if model_name in ["dfNGP", "dfNN", "PINN"]:
    from configs import BATCH_SIZE
    from torch.utils.data import DataLoader, TensorDataset

if model_name == "PINN":
    from NN_models import PINN_backbone
    from configs import W_PINN_DIV_WEIGHT

# universals 
from metrics import compute_RMSE, compute_MAE, compute_divergence_field

# basics
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# utilitarian
from utils import set_seed
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

#############################
### LOOP 1 - over REGIONS ###
#############################

# For region_name in ["regiona", "regionb", "regionc"]:
for region_name in ["regionc"]:

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

    # train
    x_train = train[:, [0, 1]].to(device)
    y_train = train[:, [3, 4]].to(device)

    # test
    x_test = test[:, [0, 1]].to(device)
    y_test = test[:, [3, 4]].to(device)

    # local measurment errors as noise
    train_noise_diag = torch.concat((train[:, 5], train[:, 6]), dim = 0).to(device)

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

    for run in range(NUM_RUNS):

        print(f"\n--- Training Run {run + 1}/{NUM_RUNS} ---")

        # convert to DataLoader for batching
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

        # initialise new model for run (seeded so this is reproducible)
        PINN_model = PINN_backbone().to(device)
        PINN_model.train()

        # define loss function (MSE for regression)
        criterion = torch.nn.MSELoss()

        # AdamW as optimizer for some regularisation/weight decay
        optimizer = optim.AdamW(PINN_model.parameters(), lr = MODEL_LEARNING_RATE, weight_decay = WEIGHT_DECAY)

        # _________________
        # BEFORE EPOCH LOOP
        
        # Export the convergence just for first run only
        if run == 0:
            # initialise tensors to store losses over epochs (for convergence plot)
            train_losses_PINN_over_epochs = torch.zeros(MAX_NUM_EPOCHS) # objective
            train_losses_RMSE_over_epochs = torch.zeros(MAX_NUM_EPOCHS) # by-product
            test_losses_PINN_over_epochs = torch.zeros(MAX_NUM_EPOCHS)
            test_losses_RMSE_over_epochs = torch.zeros(MAX_NUM_EPOCHS)

        # Early stopping variables
        best_loss = float('inf')
        # counter starts at 0
        epochs_no_improve = 0

        ############################
        ### LOOP 3 - over EPOCHS ###
        ############################
        print("\nStart Training")

        for epoch in range(MAX_NUM_EPOCHS):

            # accumulate losses over batches for each epoch
            train_losses_PINN_over_batches = 0.0  
            train_losses_PMSE_over_batches = 0.0

            #############################
            ### LOOP 4 - over BATCHES ###
            #############################

            for batch in dataloader:

                # set model to training mode
                PINN_model.train()

                x_batch, y_batch = batch
                # put on GPU if available
                # NOTE: requires_grad_() is used to compute gradients for the input
                x_batch, y_batch = x_batch.to(device).requires_grad_(), y_batch.to(device)

                # Forward pass
                # NOTE: We used to do this with vmaps, but now we do it with the model directly (not faster)
                y_pred_batch = PINN_model(x_batch)

                # Compute divergence for the loss term
                # NOTE: We take the abs() divergence to account for negative divergence and mean() to get the scalar
                batch_divergence_loss = compute_divergence_field(y_pred_batch, x_batch).abs().mean().item()

                # Compute loss as weighted sum: (RMSE for same units as data) + divergence loss
                loss = (1 - W_PINN_DIV_WEIGHT) * torch.sqrt(criterion(y_pred_batch, y_batch)) + W_PINN_DIV_WEIGHT * batch_divergence_loss

                # Add losses to the epoch loss (over batches)
                train_losses_PINN_over_batches += loss.item()
                train_losses_PMSE_over_batches += torch.sqrt(criterion(y_pred_batch, y_batch)).item()

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ###############################
            ### END LOOP 4 over BATCHES ###
            ###############################

            # for every epoch...

            PINN_model.eval()

            # Compute average loss for the epoch (e.g. 7 batches / epoch)
            avg_train_loss_PINN_for_epoch = train_losses_PINN_over_batches / len(dataloader)

            # Print for epoch
            print(f"{region_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training PINN Loss (RMSE + div): {avg_train_loss_PINN_for_epoch:.4f}")

            # Early stopping check
            if avg_train_loss_PINN_for_epoch < best_loss:
                best_loss = avg_train_loss_PINN_for_epoch
                epochs_no_improve = 0  # reset counter
                best_model_state = PINN_model.state_dict()  # save best model
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

            # For Run 1 we save a bunch of metrics, while for the rest we only update (above)
            if run == 0:

                # Train 
                # NOTE: We do this again because we want to pass through the full dataset, not just batches
                y_train_pred = PINN_model(x_train.to(device).requires_grad_())
                # Compute train loss for loss convergence plot
                train_rmse_loss = torch.sqrt(criterion(y_train_pred, y_train.to(device))).item()
                train_PINN_loss = (1 - W_PINN_DIV_WEIGHT) * train_rmse_loss + W_PINN_DIV_WEIGHT * compute_divergence_field(y_train_pred, x_train).abs().mean().item()
                # TODO: Maybe detach here

                # Test No batches, but full dataset
                y_test_pred = PINN_model(x_test.to(device).requires_grad_())
                test_rmse_loss = torch.sqrt(criterion(y_test_pred, y_test.to(device))).item()
                test_PINN_loss = (1 - W_PINN_DIV_WEIGHT) * test_rmse_loss + W_PINN_DIV_WEIGHT * compute_divergence_field(y_test_pred, x_test).abs().mean().item()

                train_losses_RMSE_over_epochs[epoch] = train_rmse_loss
                test_losses_RMSE_over_epochs[epoch] = test_rmse_loss
                train_losses_PINN_over_epochs[epoch] = train_PINN_loss
                test_losses_PINN_over_epochs[epoch] = test_PINN_loss

        ##############################
        ### END LOOP 3 over EPOCHS ###
        ##############################

        # for every run...
        ############################################################
        ### EVALUATE after all training for this RUN is finished ###
        ############################################################

        # Load the best model for this "run"
        PINN_model.load_state_dict(best_model_state)

        # Announce what we are doing
        print(f"Run {run + 1}/{NUM_RUNS}, Training of {model_name} complete for {region_name.upper()}. Restored best model.")

        # Evaluate the trained model after epochs are finished
        PINN_model.eval()

        # For "metrics_per_run" and "metrics_summary" we need to compute the divergence field
        # turn on gradient tracking for divergence
        x_train_grad = x_train.to(device).requires_grad_()
        x_test_grad = x_test.to(device).requires_grad_()

        y_train_PINN_predicted = PINN_model(x_train_grad)
        y_test_PINN_predicted = PINN_model(x_test_grad)

        PINN_train_div_field = compute_divergence_field(y_train_PINN_predicted, x_train_grad).detach()
        PINN_test_div_field = compute_divergence_field(y_test_PINN_predicted, x_test_grad).detach()

        PINN_train_MAD = PINN_train_div_field.abs().mean().item()
        PINN_test_MAD = PINN_test_div_field.abs().mean().item()

        # Compute metrics (convert tensors to float)
        PINN_train_RMSE = compute_RMSE(y_train, y_train_PINN_predicted).item()
        PINN_train_MAE = compute_MAE(y_train, y_train_PINN_predicted).item()

        PINN_test_RMSE = compute_RMSE(y_test, y_test_PINN_predicted).item()
        PINN_test_MAE = compute_MAE(y_test, y_test_PINN_predicted).item()

        # Store results of best model for run[i] in list
        region_results.append([
            run + 1, 
            PINN_train_RMSE, PINN_train_MAE, PINN_train_MAD,
            PINN_test_RMSE, PINN_test_MAE, PINN_test_MAD
        ])

        # For first run only, we save the predictions, div field and loss evolution over epochs
        if run == 0:
            # (1) Save predictions from first run so we can visualise them later
            torch.save(y_test_PINN_predicted, f"{MODEL_REAL_RESULTS_DIR}/{region_name}_{model_name}_test_predictions.pt")

            # (2) Save divergence field over test
            # NOTE: The test set is not a full field, but a subset of the field
            torch.save(PINN_test_div_field, f"{MODEL_REAL_RESULTS_DIR}/{region_name}_{model_name}_test_prediction_divergence_field.pt")

            # (3) Save "losses_over_epochs" for run 1
            df_losses = pd.DataFrame({
                'Epoch': list(range(train_losses_RMSE_over_epochs.shape[0])), # pythonic indexing
                'Train Loss RMSE': train_losses_RMSE_over_epochs.tolist(), 
                'Train Loss PINN': train_losses_PINN_over_epochs.tolist(),
                'Test Loss RMSE': test_losses_RMSE_over_epochs.tolist(),
                'Test Loss PINN': test_losses_PINN_over_epochs.tolist()
                })
            
            df_losses.to_csv(f"{MODEL_REAL_RESULTS_DIR}/{region_name}_{model_name}_losses_over_epochs.csv", index = False, float_format = "%.5f")

        # Free up memory at end of each run
        del PINN_model, y_train_PINN_predicted, y_test_PINN_predicted, PINN_train_div_field, PINN_test_div_field, PINN_train_MAD, PINN_test_MAD, PINN_test_MAE, PINN_test_RMSE, PINN_train_MAE, PINN_train_RMSE, train_losses_PINN_over_batches, train_losses_PMSE_over_batches, train_losses_RMSE_over_epochs, test_losses_RMSE_over_epochs, train_losses_PINN_over_epochs, test_losses_PINN_over_epochs

        # Call garbage collector to free up memory
        gc.collect()
        torch.cuda.empty_cache()
        
    ############################
    ### END LOOP 2 over RUNS ###
    ############################

    # Convert results to a Pandas DataFrame
    results_per_run = pd.DataFrame(
        region_results, 
        columns = ["Run", 
                   "Train RMSE", "Train MAE", "Train MAD",
                   "Test RMSE", "Test MAE", "Test MAD"])

    # Compute mean and standard deviation for each metric
    mean_std_df = results_per_run.iloc[:, 1:].agg(["mean", "std"]) # Exclude "Run" column

    # Add region_name and model_name as columns in the DataFrame _metrics_summary to be able to copy df
    mean_std_df["region name"] = region_name
    mean_std_df["model name"] = model_name

    # Save "_metrics_per_run.csv" to CSV
    path_to_metrics_per_run = os.path.join(MODEL_REAL_RESULTS_DIR, f"{region_name}_{model_name}_metrics_per_run.csv")
    results_per_run.to_csv(path_to_metrics_per_run, index = False, float_format = "%.5f") # reduce to 5 decimals
    print(f"\nResults per run saved to {path_to_metrics_per_run}")

    # Save "_metrics_summary.csv" to CSV
    path_to_metrics_summary = os.path.join(MODEL_REAL_RESULTS_DIR, f"{region_name}_{model_name}_metrics_summary.csv")
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

if device == "cuda":
    # get name of GPU model
    gpu_name = torch.cuda.get_device_name(0)
else:
    gpu_name = "N/A"

print(f"Elapsed wall time: {elapsed_time:.4f} seconds")

# Define full path for the file
wall_time_and_gpu_path = os.path.join(MODEL_REAL_RESULTS_DIR, model_name + "_run_" "wall_time.txt")

# Save to the correct folder with both seconds and minutes
with open(wall_time_and_gpu_path, "w") as f:
    f.write(f"Elapsed wall time: {elapsed_time:.4f} seconds\n")
    f.write(f"Elapsed wall time: {elapsed_time_minutes:.2f} minutes\n")
    f.write(f"Device used: {device}\n")
    f.write(f"GPU model: {gpu_name}\n")

print(f"Wall time saved to {wall_time_and_gpu_path}.")