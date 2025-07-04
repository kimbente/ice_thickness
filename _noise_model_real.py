  # NOTE: Here we estimate the noise variance 
    """
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
    """