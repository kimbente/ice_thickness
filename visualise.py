
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

############
### CMAP ###
############

# Generate Berlin colormap although it is only available in Matplotlib 3.10
# Load tensor for berlin (dark diverging cmap) from folder
cmap_berlin_tensor = torch.load("configs/vis/cmap_berlin_tensor.pt", weights_only = True)

# convert to list
_berlin_data = cmap_berlin_tensor.tolist()

from matplotlib.colors import ListedColormap

cmaps = {
    name: ListedColormap(data, name = name) for name, data in [
        ('berlin', _berlin_data),
    ]}

cmaps['berlin']

##############
### Quiver ###
##############

def visualise_v_quiver(v, x, 
                       div_v = None, title_string = "v(x)", color_abs_max = 0.5, lw_scalar = 2):
    """Plots a vector field v(x) and its divergence div_v(x) as a quiverplot on a square grid.
    The quiverlength automatically corresponds to the magnitude/speed of the vector field.
    The color corresponds to the divergence of the vector field. We use the dark diverging colormap "berlin" so that zero divergence is visible as black. 

    Args:
        v (torch.Size([N_long, 2])): flattened square vector field, where the first column is the u component and the second column is the v component
        x (torch.Size([N_long, 2])): flattend meshgrids, where the first column is the x component and the second column is the y component
        div_v (torch.Size([N_long], optional)): flat divergence of the vector field,
        title_string (str, optional): Title for plot. Defaults to "v(x)".
        color_abs_max (float, optional): Maximum absolute value for color normalization. Defaults to 0.5.
    """

    # Extract N_long and calculate sqrt of N_long, N_side
    N_long = torch.tensor(v.shape[0])
    N_side = int(torch.sqrt(N_long))

    # Extract both columns/components from v and make square
    U = v[:, 0].reshape(N_side, N_side)
    V = v[:, 1].reshape(N_side, N_side)

    # black color if nothing is parsed in.
    if div_v is None:
        div_v = torch.zeros_like(v[:, 0])

    # Make coordinates square again
    X = x[:, 0].reshape(N_side, N_side)
    Y = x[:, 1].reshape(N_side, N_side)

    div_v_square = div_v.reshape(N_side, N_side)
    # Define symmetric normalization with zero centered
    norm = mcolors.TwoSlopeNorm(vmin = - color_abs_max, vcenter = 0, vmax = color_abs_max)

    # Magnitude i.e. speed: square each element to remove negative direction, then square root
    # mag = torch.sqrt(torch.square(U) + torch.square(V))
    # lw = mag * lw_scalar / torch.max(mag) # normalise mag

    fig, ax = plt.subplots(1, 1, figsize = (4, 4))

    ax.quiver(X.numpy(), Y.numpy(), U.numpy(), V.numpy(),
              div_v_square.numpy(), # color is passed directly
              cmap = cmaps['berlin'],
              norm = norm)

    # ax.quiver(x[:, 0], x[:, 1], v[:, 0], v[:, 1])
    
    # coolwarm is diverging but has grey in middle
    # add norm
    ax.set_aspect(1)
    ax.set_title(title_string)

    plt.show()

##############
### STREAM ###
##############

def visualise_v_stream(v, x, div_v = None, title_string = "v(x)", color_abs_max = 0.5, lw_scalar = 2):
    """Plots a vector field v(x) and its divergence div_v(x) as a streamplot on a square grid.
    The linewidth corresponds to the magnitude/speed of the vector field.
    The color corresponds to the divergence of the vector field. We use the dark doiverging colormap "berlin" so that zero divergence is visible as black. 

    Args:
        v (torch.Size([N_long, 2])): flattened square vector field, where the first column is the u component and the second column is the v component
        div_v (torch.Size([N_long])): flat divergence of the vector field
        x (torch.Size([N_long, 2])): flattend meshgrids, where the first column is the x component and the second column is the y component
        title_string (str, optional): Title for plot. Defaults to "v(x)".
        color_abs_max (float, optional): Maximum absolute value for color normalization. Defaults to 0.5.
    """

    # Extract N_long and calculate sqrt of N_long, N_side
    N_long = torch.tensor(v.shape[0])
    N_side = int(torch.sqrt(N_long))

    # Extract both columns/components from v and make square
    U = v[:, 0].reshape(N_side, N_side)
    V = v[:, 1].reshape(N_side, N_side)

    # Make coordinates square again
    X = x[:, 0].reshape(N_side, N_side)
    Y = x[:, 1].reshape(N_side, N_side)

    # black color if nothing is parsed in.
    if div_v is None:
        div_v = torch.zeros_like(v[:, 0])

    div_v_square = div_v.reshape(N_side, N_side)
    # Define symmetric normalization with zero centered
    norm = mcolors.TwoSlopeNorm(vmin = - color_abs_max, vcenter = 0, vmax = color_abs_max)

    # Magnitude i.e. speed: square each element to remove negative direction, then square root
    mag = torch.sqrt(torch.square(U) + torch.square(V))
    lw = mag * lw_scalar / torch.max(mag) # normalise mag

    fig, ax = plt.subplots(1, 1, figsize = (4, 4))

    ax.streamplot(X.numpy(), Y.numpy(), U.numpy(), V.numpy(), 
                  linewidth = lw.numpy(),
                  color = div_v_square.numpy(), 
                  cmap = cmaps['berlin'],
                  norm = norm)
    
    # coolwarm is diverging but has grey in middle
    # add norm
    ax.set_aspect(1)
    ax.set_title(title_string)

    plt.show()