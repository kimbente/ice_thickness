from GP_models import GP_predict
from metrics import compute_RMSE, compute_MAE, compute_NLL, compute_NLL_full
from utils import set_seed

# Global file for training configs
from configs import PATIENCE, MAX_NUM_EPOCHS, NUM_RUNS, GP_LEARNING_RATE, WEIGHT_DECAY, SIGMA_N_RANGE, SIGMA_F_RANGE, L_RANGE, dfGP_REAL_RESULTS_DIR
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import gc

# Set seed for reproducibility
set_seed(42)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# device = 'cpu'

model_name = "dfGP"

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
    # MAX_NUM_EPOCHS = 400

    # Number of training runs for mean and std of metrics
    NUM_RUNS = NUM_RUNS
    NUM_RUNS = 1
    LEARNING_RATE = GP_LEARNING_RATE
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = WEIGHT_DECAY

    # Pass in all the training data for GPs
    # Don't need dataloader either
    # BATCH_SIZE = BATCH_SIZE

    # Ensure the results folder exists
    RESULTS_DIR = dfGP_REAL_RESULTS_DIR
    os.makedirs(RESULTS_DIR, exist_ok = True)
    ### LOOP OVER RUNS ###
    for run in range(NUM_RUNS):
        print(f"\n--- Training Run {run + 1}/{NUM_RUNS} ---")

        # Sample from uniform distributions to initialise hyperparameters
        # We could inform this 
        # sigma_n = torch.tensor([0.05], requires_grad = False).to(device) # no optimisation for noise, no sampling
        # sigma_n = nn.Parameter(torch.empty(1, device = device).uniform_( * SIGMA_N_RANGE)) # Not Trainable
        sigma_n = torch.tensor([0.005], requires_grad = False).to(device)

        sigma_f = nn.Parameter(torch.empty(1, device = device).uniform_( * SIGMA_F_RANGE)) # Trainable
        # Initialising l from a uniform distribution as nn.Param to avoid leaf variable error
        l = nn.Parameter(torch.empty(2, device = device).uniform_( * L_RANGE))
        
        # We do not need to "initialse" the GP model
        # We don't need a criterion either

        # Define optimizer (e.g., AdamW)
        optimizer = optim.AdamW([sigma_f, l], lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

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

        # We do not need to add extra noise to real data

        ### LOOP OVER EPOCHS ###
        print("\nStart Training")
        for epoch in range(MAX_NUM_EPOCHS):

            # No batching - full epoch pass in one
            if run == 0:
                mean_pred_train, _, lml_train = GP_predict(
                        x_train,
                        y_train,
                        x_train, # have predictions for training data again
                        [sigma_n, sigma_f, l], # initial hyperparameters
                        # no mean
                        divergence_free_bool = True,
                        train_noise_input = train_noise_diagmatrix)
                
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
                        [sigma_n, sigma_f, l], # initial hyperparameters
                        # no mean
                        divergence_free_bool = True,
                        train_noise_input = train_noise_diagmatrix)
                
                train_RMSE = compute_RMSE(y_train, mean_pred_train)
                test_RMSE = compute_RMSE(y_test, mean_pred_test)

                gc.collect() and torch.cuda.empty_cache()

                epoch_train_NLML_losses[epoch] = - lml_train
                epoch_train_RMSE_losses[epoch] = train_RMSE
                # epoch_test_NLML_losses[epoch] =  # train NLML
                epoch_test_RMSE_losses[epoch] = test_RMSE

                epoch_sigma_f[epoch] = sigma_f[0]
                epoch_l1[epoch] = l[0]
                epoch_l2[epoch] = l[1]

                print(f"{region_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training Loss (NLML): {loss:.4f}, (RMSE): {train_RMSE:.4f}")
            
            else:
                # Save compute after run 1
                _, _, lml_train = GP_predict(
                        x_train,
                        y_train,
                        x_train[0:2], # have predictions for training data again
                        [sigma_n, sigma_f, l], # initial hyperparameters
                        # no mean
                        divergence_free_bool = True, 
                        train_noise_input = train_noise_diagmatrix)
                
                gc.collect() and torch.cuda.empty_cache()
                
                loss = - lml_train
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"{region_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training Loss (NLML): {loss:.4f}")

            # Early stopping check
            if loss < best_loss:
                best_loss = loss
                epochs_no_improve = 0  # Reset counter
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

            # After every epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                     [sigma_n, sigma_f, l], # optimal hypers
                     # no mean
                     divergence_free_bool = True,
                     train_noise_input = train_noise_diagmatrix)

        # Only save things for one run
        if run == 0:
            #(1) Save predictions from first run so we can visualise them later
            torch.save(mean_pred_test, f"{RESULTS_DIR}/{region_name}_{model_name}_test_mean_predictions.pt")
            torch.save(covar_pred_test, f"{RESULTS_DIR}/{region_name}_{model_name}_test_covar_predictions.pt")

            #(2) Save best hyperparameters from run 1
            # Stack tensors into a single tensor
            best_hypers_tensor = torch.cat([
                sigma_n.reshape(-1),  # Ensure 1D shape
                sigma_f.reshape(-1),
                l.reshape(-1)
            ])

            # Save the tensor
            torch.save(best_hypers_tensor, f"{RESULTS_DIR}/{region_name}_{model_name}_best_hypers.pt")

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
            
            df_losses.to_csv(f"{RESULTS_DIR}/{region_name}_{model_name}_losses_over_epochs.csv", index = False, float_format = "%.5f") # reduce to 5 decimals

            # #(4) Save divergence field
            u_indicator_test, v_indicator_test = torch.zeros_like(mean_pred_test), torch.zeros_like(mean_pred_test)
            u_indicator_test[:, 0] = 1.0 # output column u selected
            v_indicator_test[:, 1] = 1.0 # output column v selected

            # divergence field (positive and negative divergences)
            dfGP_test_div_field = (torch.autograd.grad(
                outputs = mean_pred_test,
                inputs = x_test_grad,
                grad_outputs = u_indicator_test,
                create_graph = True
            )[0][:, 0] + torch.autograd.grad(
                outputs = mean_pred_test,
                inputs = x_test_grad,
                grad_outputs = v_indicator_test,
                create_graph = True
            )[0][:, 1])

            # Save as test predition divergence field
            torch.save(dfGP_test_div_field, f"{RESULTS_DIR}/{region_name}_{model_name}_test_prediction_divergence_field.pt")

        x_train_grad = x_train.to(device).requires_grad_(True)

        mean_pred_train, covar_pred_train, _ = GP_predict(
                     x_train,
                     y_train,
                     x_train_grad,
                     [sigma_n, sigma_f, l], # optimal hypers
                     # no mean
                     divergence_free_bool = True, 
                     train_noise_input = train_noise_diagmatrix)

        ### Divergence: Total absolute divergence (sum divergence at each point, after summing dims)
        # autograd div test
        u_indicator_test, v_indicator_test = torch.zeros_like(mean_pred_test), torch.zeros_like(mean_pred_test)
        u_indicator_test[:, 0] = 1.0 # output column u selected
        v_indicator_test[:, 1] = 1.0 # output column v selected

        dfGP_test_div = (torch.autograd.grad(
            outputs = mean_pred_test,
            inputs = x_test_grad,
            grad_outputs = u_indicator_test,
            create_graph = True
        )[0][:, 0] + torch.autograd.grad(
            outputs = mean_pred_test,
            inputs = x_test_grad,
            grad_outputs = v_indicator_test,
            create_graph = True
        )[0][:, 1]).abs().mean().item() # v with respect to y

        # autograd div train
        u_indicator_train, v_indicator_train = torch.zeros_like(mean_pred_train), torch.zeros_like(mean_pred_train)
        u_indicator_train[:, 0] = 1.0 # output column u selected
        v_indicator_train[:, 1] = 1.0 # output column v selected

        dfGP_train_div = (torch.autograd.grad(
            outputs = mean_pred_train,
            inputs = x_train_grad,
            grad_outputs = u_indicator_train,
            create_graph = True
        )[0][:, 0] + torch.autograd.grad(
            outputs = mean_pred_train,
            inputs = x_train_grad,
            grad_outputs = v_indicator_train,
            create_graph = True
        )[0][:, 1]).abs().mean().item() # v with respect to y

        # Compute metrics (convert tensors to float) for every run's tuned model
        dfGP_train_RMSE = compute_RMSE(y_train, mean_pred_train).item()
        dfGP_train_MAE = compute_MAE(y_train, mean_pred_train).item()
        dfGP_train_NLL = compute_NLL(y_train, mean_pred_train, covar_pred_train).item()

        dfGP_test_RMSE = compute_RMSE(y_test, mean_pred_test).item()
        dfGP_test_MAE = compute_MAE(y_test, mean_pred_test).item()
        # full NLL has caused instability issues due to the logdet
        # now we use sparse
        dfGP_test_NLL = compute_NLL(y_test, mean_pred_test, covar_pred_test).item()

        region_results.append([
            run + 1,
            dfGP_train_RMSE, dfGP_train_MAE, dfGP_train_NLL, dfGP_train_div,
            dfGP_test_RMSE, dfGP_test_MAE, dfGP_test_NLL, dfGP_test_div
        ])

        del mean_pred_train, mean_pred_test, covar_pred_train, covar_pred_test
        gc.collect()
        gc.collect() and torch.cuda.empty_cache()

    ### FINISH LOOP OVER RUNS ###
    # Convert results to a Pandas DataFrame
    df = pd.DataFrame(
        region_results, 
        columns = ["Run", 
                   "Train RMSE", "Train MAE", "Train NLL", "Train MAD",
                   "Test RMSE", "Test MAE", "Test NLL", "Test MAD"])

    # Compute mean and standard deviation for each metric
    mean_std_df = df.iloc[:, 1:].agg(["mean", "std"])  # Exclude "Run" column

    # Add region_name and model_name as columns in the DataFrame _metrics_summary
    mean_std_df["region name"] = region_name
    mean_std_df["model name"] = model_name

    # Save results to CSV
    results_file = os.path.join(RESULTS_DIR, f"{region_name}_{model_name}_metrics_per_run.csv")
    df.to_csv(results_file, index = False, float_format = "%.5f") # reduce to 5 decimals
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

print(torch.cuda.memory_summary(device = None, abbreviated = False))

import torch
import gc

# Get all tensors currently tracked by Python
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj):
            if obj.is_cuda:
                print(f"Tensor on GPU: shape={obj.shape}, dtype={obj.dtype}, device={obj.device}")
        elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
            if obj.data.is_cuda:
                print(f"Tensor with data on GPU: shape={obj.data.shape}, dtype={obj.data.dtype}, device={obj.data.device}")
    except Exception as e:
        pass  # Some objects might throw errors

