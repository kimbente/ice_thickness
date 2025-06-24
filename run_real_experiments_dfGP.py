# REAL DATA EXPERIMENTS
# RUN WITH python run_real_experiments_dfGP.py
#               _                 _   _      
#              | |               | | (_)     
#    __ _ _ __ | |_ __ _ _ __ ___| |_ _  ___ 
#   / _` | '_ \| __/ _` | '__/ __| __| |/ __|
#  | (_| | | | | || (_| | | | (__| |_| | (__ 
#   \__,_|_| |_|\__\__,_|_|  \___|\__|_|\___|
# 
model_name = "dfGP"
from gpytorch_models import dfGP

# import configs to we can access the hypers with getattr
import configs
from configs import PATIENCE, MAX_NUM_EPOCHS, NUM_RUNS, WEIGHT_DECAY
from configs import TRACK_EMISSIONS_BOOL

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
    tracker = EmissionsTracker(project_name = "dfGP_real_experiments", output_dir = MODEL_REAL_RESULTS_DIR)
    tracker.start()

#############################
### LOOP 1 - over REGIONS ###
#############################

for region_name in ["region_lower_byrd", "region_mid_byrd", "region_upper_byrd"]:

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

    ### NOISE MODEL ###
    # TRAIN
    # noise variance (h * sigma_u)^2 and (h * sigma_v)^2 (tensor contains [h sig_u, h sig_v] stds)
    noise_var_h_times_uv_train = torch.concat((train[:, 5], train[:, 6]), dim = 0)**2
    # assume age dependent noise sigma_h on ice thickness measurements: ~10 - 20 m std (1000 scaling)
    sigma_h = 0.01 * torch.log(train[:, 7] + 3)
    # calculate noise variance (u * sigma_h)^2 and (v * sigma_h)^2
    noise_var_uv_times_h_train = (torch.concat((train[:, 3], train[:, 4]), dim = 0) * torch.cat([sigma_h, sigma_h]))**2
    # combine both noise variances into the std for each dimension
    train_noise_diag = torch.sqrt(noise_var_h_times_uv_train + noise_var_uv_times_h_train).to(device)

    # Compute midpoint
    midpoint = train_noise_diag.shape[0] // 2

    # Print noise levels for train, formatted to 4 decimal places
    print(f"Mean noise std per x dimension: {train_noise_diag[:midpoint].mean(dim = 0).item():.4f}")
    print(f"Mean noise std per y dimension: {train_noise_diag[midpoint:].mean(dim = 0).item():.4f}")

    # TEST
    # noise variance (h * sigma_u)^2 and (h * sigma_v)^2 (tensor contains [h sig_u, h sig_v] stds)
    noise_var_h_times_uv_test = torch.concat((test[:, 5], test[:, 6]), dim = 0)**2
    # assume age dependent noise sigma_h on ice thickness measurements: ~10 - 20 m std (1000 scaling)
    sigma_h = 0.01 * torch.log(test[:, 7] + 3)
    # calculate noise variance (u * sigma_h)^2 and (v * sigma_h)^2
    noise_var_uv_times_h_test = (torch.concat((test[:, 3], test[:, 4]), dim = 0) * torch.cat([sigma_h, sigma_h]))**2
    # combine both noise variances into the std for each dimension
    test_noise_diag = torch.sqrt(noise_var_h_times_uv_test + noise_var_uv_times_h_test).to(device)

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
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        # Intialise fresh GP model with flat x_train and y_train_noisy (block-flat)
        model = dfGP(
            x_train.T.reshape(-1),
            y_train.T.reshape(-1), 
            likelihood
            ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr = MODEL_LEARNING_RATE, weight_decay = WEIGHT_DECAY)
        
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

            # NOTE: No sigma_n
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
            loss = - mll(train_pred_dist, y_train.T.reshape(-1).to(device))  # negative marginal log likelihood
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

                # Save losses for convergence plot
                train_losses_NLML_over_epochs[epoch] = loss.item()
                train_losses_RMSE_over_epochs[epoch] = train_RMSE.item()
                test_losses_RMSE_over_epochs[epoch] = test_RMSE.item()

                # Save evolution of hypers for convergence plot
                sigma_f_over_epochs[epoch] = model.covar_module.outputscale.item()
                l1_over_epochs[epoch] = model.covar_module.lengthscale[0].item()
                l2_over_epochs[epoch] = model.covar_module.lengthscale[1].item()

                # Print a bit more information for the first run
                if epoch % 20 == 0:
                    print(f"{region_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training Loss (NLML): {loss:.4f}, (RMSE): {train_RMSE:.4f}")

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
                    print(f"{region_name} {model_name} Run {run + 1}/{NUM_RUNS}, Epoch {epoch + 1}/{MAX_NUM_EPOCHS}, Training Loss (NLML): {loss:.4f}")
                
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
        best_sigma_f = sigma_f.detach().clone()
        best_l = l.detach().clone()

        # Need gradients for autograd divergence: We clone and detach
        x_test_grad = x_test.to(device).clone().requires_grad_(True)

        mean_pred_test, covar_pred_test, _ = GP_predict(
            x_train,
            y_train,
            x_test_grad,
            [train_noise_diag, best_sigma_f, best_l], # list of (initial) hypers
            mean_func = None, # no mean aka "zero-mean function"
            divergence_free_bool = True) # ensures we use a df kernel
        
        # Compute divergence field
        dfGP_test_div_field = compute_divergence_field(mean_pred_test, x_test_grad)

        # Only save mean_pred, covar_pred and divergence fields for the first run
        if run == 0:

            # (1) Save predictions from first run so we can visualise them later
            torch.save(mean_pred_test, f"{MODEL_REAL_RESULTS_DIR}/{region_name}_{model_name}_test_mean_predictions.pt")
            torch.save(covar_pred_test, f"{MODEL_REAL_RESULTS_DIR}/{region_name}_{model_name}_test_covar_predictions.pt")

            # (2) Save best hyperparameters
            # Stack tensors into a single tensor
            best_hypers_tensor = torch.cat([
                best_sigma_f,
                best_l
            ])

            torch.save(best_hypers_tensor, f"{MODEL_REAL_RESULTS_DIR}/{region_name}_{model_name}_best_hypers.pt")

            # (3) Since all epoch training is finished, we can save the losses over epochs
            df_losses = pd.DataFrame({
                'Epoch': list(range(train_losses_NLML_over_epochs.shape[0])), # pythonic indexing
                'Train Loss NLML': train_losses_NLML_over_epochs.tolist(),
                'Train Loss RMSE': train_losses_RMSE_over_epochs.tolist(),
                'Test Loss RMSE': test_losses_RMSE_over_epochs.tolist(),
                'Sigma_f': sigma_f_over_epochs.tolist(),
                'l1': l1_over_epochs.tolist(),
                'l2': l2_over_epochs.tolist()
                })
            
            df_losses.to_csv(f"{MODEL_REAL_RESULTS_DIR}/{region_name}_{model_name}_losses_over_epochs.csv", index = False, float_format = "%.5f") # reduce to 5 decimals for readability

            # (4) Save divergence field (computed above for all runs)
            torch.save(dfGP_test_div_field, f"{MODEL_REAL_RESULTS_DIR}/{region_name}_{model_name}_test_prediction_divergence_field.pt")

        x_train_grad = x_train.to(device).clone().requires_grad_(True)

        mean_pred_train, covar_pred_train, _ = GP_predict(
                     x_train,
                     y_train,
                     x_train_grad,
                     [train_noise_diag, best_sigma_f, best_l], # list of (initial) hypers
                     mean_func = None, # no mean aka "zero-mean function"
                     divergence_free_bool = True) # ensures we use a df kernel
        
        dfGP_train_div_field = compute_divergence_field(mean_pred_train, x_train_grad)

        # Divergence: Convert field to metric: mean absolute divergence
        # NOTE: It is important to use the absolute value of the divergence field, since positive and negative deviations are violations and shouldn't cancel each other out 
        dfGP_train_div = dfGP_train_div_field.abs().mean().item()
        dfGP_test_div = dfGP_test_div_field.abs().mean().item()

        # Compute metrics (convert tensors to float) for every run's tuned model
        dfGP_train_RMSE = compute_RMSE(y_train, mean_pred_train).item()
        dfGP_train_MAE = compute_MAE(y_train, mean_pred_train).item()
        dfGP_train_sparse_NLL = compute_NLL_sparse(y_train, mean_pred_train, (covar_pred_train + torch.diag(train_noise_diag**2))).item()
        dfGP_train_full_NLL, dfGP_train_jitter = compute_NLL_full(y_train, mean_pred_train, (covar_pred_train + torch.diag(train_noise_diag**2)))

        dfGP_test_RMSE = compute_RMSE(y_test, mean_pred_test).item()
        dfGP_test_MAE = compute_MAE(y_test, mean_pred_test).item()
        dfGP_test_sparse_NLL = compute_NLL_sparse(y_test, mean_pred_test, (covar_pred_test + torch.diag(test_noise_diag**2))).item()
        dfGP_test_full_NLL, dfGP_test_jitter = compute_NLL_full(y_test, mean_pred_test, (covar_pred_test + torch.diag(test_noise_diag**2)))

        region_results.append([
            run + 1,
            dfGP_train_RMSE, dfGP_train_MAE, dfGP_train_sparse_NLL, dfGP_train_full_NLL.item(), dfGP_train_jitter.item(),  dfGP_train_div,
            dfGP_test_RMSE, dfGP_test_MAE, dfGP_test_sparse_NLL, dfGP_test_full_NLL.item(), dfGP_test_jitter.item(), dfGP_test_div
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
        region_results, 
        columns = ["Run", 
                   "Train RMSE", "Train MAE", "Train sparse NLL", "Train full NLL", "Train jitter", "Train MAD",
                   "Test RMSE", "Test MAE", "Test sparse NLL", "Test full NLL", "Test jitter", "Test MAD"])

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
wall_time_and_gpu_path = os.path.join(MODEL_REAL_RESULTS_DIR, model_name + "_run_" "wall_time.txt")

# Save to the correct folder with both seconds and minutes
with open(wall_time_and_gpu_path, "w") as f:
    f.write(f"Elapsed wall time: {elapsed_time:.4f} seconds\n")
    f.write(f"Elapsed wall time: {elapsed_time_minutes:.2f} minutes\n")
    f.write(f"Device used: {device}\n")
    f.write(f"GPU model: {gpu_name}\n")

print(f"Wall time saved to {wall_time_and_gpu_path}.")