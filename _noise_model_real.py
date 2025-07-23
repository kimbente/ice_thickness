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

   # NOTE: Here we estimate the noise variance, but use a domain-informed prior
    ### NOISE MODEL ###
    # Thickness^2 * error_uv^2 (t^2 * sigma_u^2, t^2 * sigma_v^2)
    ### NOISE MODEL ###
    # Thickness^2 * error_uv^2 (t^2 * sigma_u^2, t^2 * sigma_v^2)
    noise_var_t_sq_times_uv_var = torch.cat([
        (train[:, 9]**2 * train[:, 5]**2),
        (train[:, 9]**2 * train[:, 6]**2),
    ], dim = 0)

    # UV^2 * error_thickness^2 (u^2 * sigma_t^2, v^2 * sigma_t^2)
    # Calculate the factor for sigma_t (indpendent of scaling)
    factor = 7.5 / train[:, 11]
    # noise std level: only dependent on age, not depth, abou 15 m std
    sigma_t = torch.cat([
        factor * torch.log(train[:, 10] + 3),
        factor * torch.log(train[:, 10] + 3)
    ], dim = 0)
    noise_var_uv_sq_times_t_var = (torch.concat((train[:, 7]**2, train[:, 8]**2), dim = 0) * sigma_t**2)

    # Combine via independed error propagation
    noise_var = noise_var_t_sq_times_uv_var + noise_var_uv_sq_times_t_var

    # Get quantiles for prior
    lower_noise_var = torch.quantile(noise_var, 0.05, dim = 0).item()
    upper_noise_var = torch.quantile(noise_var, 0.95, dim = 0).item()

    REAL_NOISE_VAR_RANGE = (lower_noise_var, upper_noise_var)

    print(f"Lower percentile noise var: {lower_noise_var:.4f}")
    print(f"Upper percentile noise var: {upper_noise_var:.4f}")
    print()