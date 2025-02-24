import torch

def log_likelihood_test(predictive_mean, predictive_covar, Y_test):

    ### STEP 1 ###
    # Change format of predctive covar from (N x 2, N x 2) to (N, 2, 2) so N (2, 2) matrices:
    n_test = Y_test.shape[0]

    var_y1_y1 = torch.diag(predictive_covar[:n_test, :n_test])
    var_y2_y2 = torch.diag(predictive_covar[n_test:, n_test:])

    covar_y1_y2 = torch.diag(predictive_covar[:n_test, n_test:])
    covar_y2_y1 = torch.diag(predictive_covar[n_test:, :n_test])

    col1 = torch.cat([var_y1_y1.unsqueeze(-1), covar_y1_y2.unsqueeze(-1)], dim = -1)
    col2 = torch.cat([covar_y2_y1.unsqueeze(-1), var_y2_y2.unsqueeze(-1)], dim = -1)
    
    covar_N22 = torch.cat([col1.unsqueeze(-1), col2.unsqueeze(-1)], dim = -1)

    ### STEP 2 ###
    # cal log likelihood in torch 
    # torch.Size([N, 2]) differences in y
    diff = predictive_mean - Y_test

    # batch matrix multiplication (n, 2) * (n, 2, 2) -> (n, 2)
    mahalanobis_dist = torch.mul(torch.matmul(torch.inverse(covar_N22), diff.unsqueeze(-1)).squeeze(-1), diff).sum(dim = -1)
    
    # element-wise determenant of all 2x2 matrices
    log_det_Sigma = torch.logdet(covar_N22)

    # element-wise log likelihood
    log_likelihood_tensor = -0.5 * (mahalanobis_dist + log_det_Sigma + 2 * torch.log(torch.tensor(2 * torch.pi)))

    # sum over all N (because we are in the log domain)
    log_likelihood = log_likelihood_tensor.sum()
    
    # scalar
    return log_likelihood