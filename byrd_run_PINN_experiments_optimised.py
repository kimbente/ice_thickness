# REAL DATA EXPERIMENTS
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

# imports for probabilistic models
if model_name in ["GP", "dfGP", "dfNGP"]:
    from GP_models import GP_predict
    from metrics import compute_NLL_sparse, compute_NLL_full
    from configs import L_RANGE, GP_PATIENCE
    # overwrite with GP_PATIENCE
    PATIENCE = GP_PATIENCE
    if model_name in ["dfGP", "dfNGP"]:
        from configs import SIGMA_F_RANGE
    if model_name == "GP":
        from configs import B_DIAGONAL_RANGE, B_OFFDIAGONAL_RANGE

# for all models with NN components train on batches
if model_name in ["dfNGP", "dfNN", "PINN"]:
    from configs import BATCH_SIZE

if model_name in ["dfNGP", "dfNN"]:
    from NN_models import dfNN

if model_name == "PINN":
    from NN_models import PINN_backbone
    from torch.utils.data import DataLoader, TensorDataset

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

    # Store metrics for the current region (used for metrics_summary report)
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

                PINN_model.train()

                x_batch, y_batch = batch
                # put on GPU if available
                # NOTE: requires_grad_() is used to compute gradients for the input
                x_batch, y_batch = x_batch.to(device).requires_grad_(), y_batch.to(device)

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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ### END LOOP OVER BATCHES ###

            PINN_model.eval()
            # Compute test loss for loss convergence plot
            y_test_pred = PINN_model(x_test.to(device).requires_grad_())
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
                best_model_state = PINN_model.state_dict()  # Save best model
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        ### END LOOP OVER EPOCHS FOR THIS RUN ###
        # Load the best model before stopping for this "run"
        PINN_model.load_state_dict(best_model_state)
        print(f"Run {run + 1}/{NUM_RUNS}, Training of {model_name} complete for {region_name.upper()}. Restored best model.")

        ################
        ### EVALUATE ###
        ################

        # Evaluate the trained model after epochs are finished
        PINN_model.eval()

        # grad for autograd
        x_train_grad = x_train.to(device).requires_grad_()
        x_test_grad = x_test.to(device).requires_grad_()

        y_train_PINN_predicted = PINN_model(x_train_grad)
        y_test_PINN_predicted = PINN_model(x_test_grad)

        # Only save things for one run
        if run == 0:
            #(1) Save predictions from first run so we can visualise them later
            torch.save(y_test_PINN_predicted, f"{RESULTS_DIR}/{region_name}_{model_name}_test_predictions.pt")

            #(2) Save loss over epochs
            df_losses = pd.DataFrame({
                'Epoch': list(range(epoch_train_losses.shape[0])), # pythonic
                'Train Loss RMSE': epoch_train_losses.tolist(), 
                'Test Loss RMSE': epoch_test_losses.tolist()
                })
            
            df_losses.to_csv(f"{RESULTS_DIR}/{region_name}_{model_name}_losses_over_epochs.csv", index = False, float_format = "%.5f")

            # No pretraining done

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
            torch.save(PINN_test_div_field, f"{RESULTS_DIR}/{region_name}_{model_name}_test_prediction_divergence_field.pt")

        # autograd divergence train
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

        # Divergence
        # autograd divergence test
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
        PINN_train_RMSE = compute_RMSE(y_train, y_train_PINN_predicted).item()
        PINN_train_MAE = compute_MAE(y_train, y_train_PINN_predicted).item()

        PINN_test_RMSE = compute_RMSE(y_test, y_test_PINN_predicted).item()
        PINN_test_MAE = compute_MAE(y_test, y_test_PINN_predicted).item()

        # Store results in list
        region_results.append([
            run + 1, PINN_train_RMSE, PINN_train_MAE, PINN_train_div,
            PINN_test_RMSE, PINN_test_MAE, PINN_test_div
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
    mean_std_df["region name"] = region_name
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