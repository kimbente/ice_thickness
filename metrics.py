import torch
import gpytorch

########################################
### QUANTILE COVERAGE ERROR (QCE) 2D ###
########################################

# NOTE: We use this for all Bayeisan models
def quantile_coverage_error_2d(
    pred_dist: gpytorch.distributions.MultitaskMultivariateNormal,
    test_y: torch.Tensor,
    quantile: float = 95.0,
):
    """
    Quantile Coverage Error for 2D multitask outputs.
    
    Args:
        pred_dist (MultitaskMultivariateNormal): GP predictions for two tasks
        test_y (torch.Tensor): Ground truth tensor of shape (N, T), where T is number of tasks.
        quantile (float): Desired quantile coverage (e.g., 95.0 for 95% CI).
        # HACK: not 0.95 but 95.0 to match the rest of the code

    Returns:
        torch.Tensor: Scalar quantile coverage error.
    """
    if quantile <= 1 or quantile >= 100:
        # NOTE: We assume that no one wants to calculate a quantile < 1 and that this an input format error.
        raise ValueError("Quantile must be between 1 and 100")

    # Flatten prediction and ground truth
    pred_mean = pred_dist.mean  # shape: (N * T,)
    pred_std = pred_dist.stddev  # shape: (N * T,)

    # Compute quantile deviation (e.g., 1.96 for 95%)
    # NOTE: 95% of the data should be within [pred_mean - deviation * pred_std, pred_mean + deviation * pred_std]
    standard_normal = torch.distributions.Normal(loc = 0.0, scale = 1.0)
    deviation = standard_normal.icdf(torch.tensor(0.5 + 0.5 * (quantile / 100.0), device = pred_mean.device))

    lower = pred_mean - deviation * pred_std
    upper = pred_mean + deviation * pred_std

    # NOTE: all.(dim = 1) checks if all elements in each row are within bounds
    # (test_y > lower) is (N, 2) -> (test_y > lower).all(dim = 1) is (N)
    # & assures that values are higher than the lower bound and lower than the upper bound at the same time
    within_bounds = ((test_y > lower).all(dim = 1) & (test_y < upper).all(dim = 1)).float()
    # Calculate the fraction of points within bounds (1's and 0's) where 1 means within bounds, i.e. TRUE
    fraction = within_bounds.mean()

    # Calculate the quantile coverage error
    return torch.abs(fraction - quantile / 100.0)

# --- For all GP models we use GPyTorch metrics, but for both NN models we use these ---
##############################
### Root Mean Square Error ###
##############################

def compute_RMSE(y_true, y_pred):
    # NOTE: Mean across all points and tasks before we take the square root.
    return torch.sqrt(torch.mean(torch.square(y_true - y_pred)))

############################
### Mean Absolute Error ###
############################

def compute_MAE(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

########################
### Divergence field ###
########################

def compute_divergence_field(mean_pred, x_grad):
    """Generate the divergence field from the mean prediction and the input gradient.
    The output of this function is later used to compute MAD, the mean absolute divergence, which is a measure of how much the flow field deviates from being divergence-free.

    Args:
        mean_pred (torch.Size(N, 2)): 2D vector field predictions, where N is the number of points.
        x_grad (torch.Size(N, 2)): 2D input points, where N is the number of points.

    Returns:
        torch.Size(N, 1): The div field is scalar because we add the two components
    """
    # Because autograd computes gradients of the output w.r.t. the inputs...
    # ... we specify which component of the output you want the gradient of via masking
    # mean_pred is a vector values output
    u_indicator, v_indicator = torch.zeros_like(mean_pred), torch.zeros_like(mean_pred)

    # output mask
    u_indicator[:, 0] = 1.0 # output column u selected
    v_indicator[:, 1] = 1.0 # output column v selected

    # divergence field (positive and negative divergences in case of non-divergence-free models)
    # NOTE: We can imput a whole field because it only depends on the point input
    div_field = (torch.autograd.grad(
        outputs = mean_pred,
        inputs = x_grad,
        grad_outputs = u_indicator,
        create_graph = True
        )[0][:, 0] + torch.autograd.grad(
        outputs = mean_pred,
        inputs = x_grad,
        grad_outputs = v_indicator,
        create_graph = True
        )[0][:, 1])
    
    return div_field