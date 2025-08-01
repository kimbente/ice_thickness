import xarray as xr
import pandas as pd
import numpy as np

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_and_preprocess_ice_velocity_data(file_path):
    """
    Load and preprocess ice velocity data from a NetCDF file.
    
    Inputs:
        file_path (str): Path to the NetCDF file.
        
    Returns:
        xarray.DataArray: Preprocessed ice velocity data.
    """
    # Load the NetCDF file
    # loading works with base(Python 3.8.12)
    ice_vel_all_of_antarctica = xr.open_dataset(file_path)

    # Data file is large so crop to the "broader Byrd region" to reduce memory usage
    corners_byrd_regions = pd.read_csv("data/real_data/corners_broader_byrd_region.csv")

    x_min, x_max, y_min, y_max = corners_byrd_regions.loc[corners_byrd_regions.name == "Byrd", ["x_min", "x_max", "y_min", "y_max"]].values[0]

    ice_vel_byrd = ice_vel_all_of_antarctica.sel(x = slice(x_min, x_max), y = slice(y_max, y_min)) # reverse order
    
    return ice_vel_byrd

# function
def create_flux_df_for_region(region_name, thickness_points, velocity_grid, corners_regions, subsample_rate = 20):
    """
    Create a DataFrame containing ice thickness and corresponding ice velocity data for a specified region.
    
    Parameters:
    - region_name: Name of the region to process.
    - thickness_points: DataFrame containing ice thickness points.
    - velocity_grid: xarray DataArray containing ice velocity data.
    
    Returns:
    - thickness_velocity_df: DataFrame with combined ice thickness and velocity data for the specified region.
    """
    
    # Get the coordinates of the corners of the selected region

    x_min, x_max, y_min, y_max = corners_regions.loc[corners_regions.name == region_name, ["x_min", "x_max", "y_min", "y_max"]].values[0]

    # Step 1: Crop the region
    thickness_points_region = thickness_points[
        (thickness_points["x"] > x_min) & 
        (thickness_points["x"] < x_max) & 
        (thickness_points["y"] > y_min) & 
        (thickness_points["y"] < y_max)]

    # Step 2: Subsample the data
    thickness_points_region_df = thickness_points_region[::subsample_rate]

    # Step 3: Interpolate ice velocity data at thickness point locations
    # Subset columns of interest first to speed up interpolation operation
    velocity_grid_ss = velocity_grid[["VX", "VY", "ERRX", "ERRY"]]
    # Drop the coordinates we don't need for the interpolation
    velocity_grid_ss = velocity_grid_ss.reset_coords(drop = True)

    # Interpolate ice velocity variables (VX, VY, ERRX, ERRY) at all combinations of x and y points from the selected region
    velocity_grid_interpolated = velocity_grid_ss.interp(
        y = (thickness_points_region_df["y"]), 
        x = (thickness_points_region_df["x"]), 
        method = "cubic") # smooth interpolation

    # I. Extract the diagonal values from each DataArray
    VX_diag = np.diag(velocity_grid_interpolated.VX.values)
    VY_diag = np.diag(velocity_grid_interpolated.VY.values)
    ERRX_diag = np.diag(velocity_grid_interpolated.ERRX.values)
    ERRY_diag = np.diag(velocity_grid_interpolated.ERRY.values)

    # II. Get the corresponding x and y coordinates
    x_coords = velocity_grid_interpolated.x.values
    y_coords = velocity_grid_interpolated.y.values

    # III. Create a DataFrame for the velocity points
    velocity_points_region_df = pd.DataFrame({
        "x_velocity": x_coords, # NOTE: for merging we indicate that these are the x coordinates of the ice velocity data
        "y_velocity": y_coords,
        "VX": VX_diag,
        "VY": VY_diag,
        "ERRX": ERRX_diag,
        "ERRY": ERRY_diag
    })
    # print(velocity_points_region_df)

    # Step 4: Combine thickness points and corresponding velocity points into a unified DataFrame
    thickness_velocity_df = pd.concat([
        thickness_points_region_df.reset_index(drop = True),
        velocity_points_region_df.reset_index(drop = True)],
        axis = 1) # concat along the columns

    # Check if the x and y coordinates match, and delete redundant columns 
    if (thickness_velocity_df["x"] == thickness_velocity_df["x_velocity"]).all() & (thickness_velocity_df["y"] == thickness_velocity_df["y_velocity"]).all():
        thickness_velocity_df = thickness_velocity_df.drop(columns = ["x_velocity", "y_velocity"]) # inplace column drop

    # Step 5: Compute flux
    thickness_velocity_df["xflux"] = thickness_velocity_df["VX"] * thickness_velocity_df["t"]
    thickness_velocity_df["yflux"] = thickness_velocity_df["VY"] * thickness_velocity_df["t"]


    # NOTE: Also export the one-sigma error 
    thickness_velocity_df["vel_x_err"] = thickness_velocity_df["ERRX"]
    thickness_velocity_df["vel_y_err"] = thickness_velocity_df["ERRY"]

    # Components
    thickness_velocity_df["vel_x"] = thickness_velocity_df["VX"]
    thickness_velocity_df["vel_y"] = thickness_velocity_df["VY"]
    thickness_velocity_df["t"] = thickness_velocity_df["t"]

    # Step 6: Add source_age column
    # define nominal year for surface velocity data, published 2017 (https://nsidc.org/sites/default/files/nsidc-0484-v002-userguide.pdf)   
    nominal_year = 2017

    # add new column with the year of the survey
    thickness_velocity_df["source_age"] = thickness_velocity_df["source"].str.extract(r"(\d{4})", expand = False).astype(int)
    # use the mean for Bedmap1 temporal range (only this dataset combines all older surveys)
    thickness_velocity_df.loc[thickness_velocity_df["source_age"] == 1966, "source_age"] = int(np.mean([1966, 2000])) # 1988
    # calculate difference to nominal age
    thickness_velocity_df["source_age"] = np.abs(thickness_velocity_df["source_age"] - nominal_year)

    print(f"Subsampled data shape: {thickness_velocity_df.shape}")
    return thickness_velocity_df

# NOTE: dark colors for older data and color families for agencies
source_color_dict = {
    'BEDMAP1_1966-2000_AIR_BM1.csv': "saddlebrown",
    'NASA_2011_ICEBRIDGE_AIR_BM2.csv': "darkred",
    'NASA_2013_ICEBRIDGE_AIR_BM3.csv': "red",
    'NASA_2017_ICEBRIDGE_AIR_BM3.csv': "pink",
    'UTIG_1999_SOAR-LVS-WLK_AIR_BM2.csv': "navy",
    'UTIG_2004_AGASEA_AIR_BM2.csv': "mediumblue",
    'UTIG_2009_Darwin-Hatherton_AIR_BM3.csv': "royalblue",
    'UTIG_2010_ICECAP_AIR_BM3.csv': "dodgerblue",
    'BAS_2007_AGAP_AIR_BM2.csv': "mediumslateblue",
    'UCANTERBURY_2008_Darwin-Hatherton_GRN_BM2.csv': "darkorange",
    'LDEO_2015_ROSETTA_AIR_BM3.csv': "forestgreen"}

def visualise_flux(region_df, source_color = False):
    fig, ax = plt.subplots(figsize = (8, 8))

    if source_color:
        ax.quiver(
                region_df["x"], 
                region_df["y"], 
                region_df["xflux"], 
                region_df["yflux"], 
                color = region_df["source"].map(source_color_dict))
    else:
        ax.quiver(
                region_df["x"], 
                region_df["y"], 
                region_df["xflux"], 
                region_df["yflux"], 
                color = "black")

    # Set both axes to use scientific notation with base 10^3
    formatter = ticker.FuncFormatter(lambda x, pos: f'{x*1e-3:.0f}')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Optional: label axes to clarify units
    ax.set_xlabel("x (km polar stereographic)")
    ax.set_ylabel("y (km polar stereographic)")

    ax.set_aspect('equal')
    ax.grid(True, alpha = 0.5)  # Show gridlines
    fig.show()


def visualise_flux_train_test_split(region_df, test_indices, usecmap = "turbo"):
    
    # define train indices
    all_indices = list(range(region_df.shape[0]))
    print(f"The dataset contains {len(all_indices)} points.")
    train_indices = list(set(all_indices) - set(test_indices))

    # Split the data into training and testing sets
    region_df_train = region_df.iloc[train_indices]
    region_df_test = region_df.iloc[test_indices]

    fig, ax = plt.subplots(figsize = (8, 8))

    # Train quivers in black
    ax.quiver(
            region_df_train["x"], 
            region_df_train["y"], 
            region_df_train["xflux"], 
            region_df_train["yflux"], 
            color = "grey")

    index_color = region_df_test.index.to_numpy()

    ax.quiver(
            region_df_test["x"], 
            region_df_test["y"], 
            region_df_test["xflux"], 
            region_df_test["yflux"], 
            index_color, 
            cmap = usecmap)

    # Set both axes to use scientific notation with base 10^3
    formatter = ticker.FuncFormatter(lambda x, pos: f'{x*1e-3:.0f}')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Optional: label axes to clarify units
    ax.set_xlabel("x (km polar stereographic)")
    ax.set_ylabel("y (km polar stereographic)")

    ax.set_aspect('equal')
    ax.grid(True, alpha = 0.5)  # Show gridlines

    fig.show()

def find_very_close_points(region_df, threshold):
    """
    Args:
        region_df (pd.DataFrame): DataFrame containing the region data with columns 'x' and 'y'.
        threshold (float): Distance threshold in meters to identify close points.
    Returns:
        list: Indices of points to be removed based on the distance threshold. List is sorted.
    """
    # Convert DataFrame to tensor (shape (n, d)) containing locations 
    x = torch.tensor(region_df[["x", "y"]].to_numpy())

    # Compute pairwise squared Euclidean distances
    dists = torch.cdist(x, x, p = 2)  # shape: [n, n]

    # Set self-distances to a large value like 10
    dists_no_diag = dists + torch.eye(dists.shape[0], device = dists.device) * 1e10

    # Find minimum distance (exlcuding self-distances)
    min_dist = dists_no_diag.min().item()
    print(f"Minimum pairwise distance found among points in dataset (in m): {min_dist:.5}")

    # Get index pairs of points where distance is < threshold (True i.e. 1)
    close_pairs = (dists_no_diag < threshold).nonzero(as_tuple = False)
    
    # print(f"Close input pairs (threshold {threshold}):")
    # print(close_pairs)

    # Track which indices to keep/remove
    to_remove = set()
    seen = set()

    # Greedy removal: remove one point from each pair
    for i, j in close_pairs:
        i = i.item()
        j = j.item()
        if i not in seen and j not in seen:
            # Remove the point with higher number (arbitrary heuristic)
            to_remove.add(j)
            seen.add(i)
            seen.add(j)

    print(f"Number of rows to remove: {len(to_remove)}")
    print(f"Rows to remove: {to_remove}")

    return sorted(to_remove)

def df_to_tensor(df, x_min, x_max, y_min, y_max, flux_scale, surface_scale = 1000):
    """
    Convert a DataFrame to a tensor and scale as we need.
    
    Parameters:
    - df: DataFrame containing the data
    - x_min, x_max, y_min, y_max: coordinates for normalisation
    - flux_scale: scaling factor for fluxes. We will devide by this.
    - surface_scale: scaling factor for surface. We will devide by this value.
    
    Returns:
    - Tensor with the data
    """
    # use domain boundries to normalise the x and y coordinates into [0, 1] range
    x_tensor = torch.tensor(
        ((df.x - x_min) / (x_max - x_min)).to_numpy(),
        dtype = torch.float32
    )

    y_tensor = torch.tensor(
        ((df.y - y_min) / (y_max - y_min)).to_numpy(),
        dtype = torch.float32
    )

    # scale surface by fixed scale (1000 m)
    s_tensor = torch.tensor(
        (df.s / surface_scale).to_numpy(),
        dtype = torch.float32
    )

    # scale fluxes by the flux_scale
    xflux_tensor = torch.tensor(
        (df.xflux / flux_scale).to_numpy(),
        dtype = torch.float32
    )

    yflux_tensor = torch.tensor(
        (df.yflux / flux_scale).to_numpy(),
        dtype = torch.float32
    )

    # error, also scaled by flux_scale
    velxerr_tensor = torch.tensor(
        (df.vel_x_err/ np.sqrt(flux_scale)).to_numpy(),
        dtype = torch.float32
    )

    velyerr_tensor = torch.tensor(
        (df.vel_x_err / np.sqrt(flux_scale)).to_numpy(),
        dtype = torch.float32
    )

    vel_x_tensor = torch.tensor(
        (df.vel_x / np.sqrt(flux_scale)).to_numpy(),
        dtype = torch.float32
    )

    vel_y_tensor = torch.tensor(
        (df.vel_y / np.sqrt(flux_scale)).to_numpy(),
        dtype = torch.float32
    )

    t_tensor = torch.tensor(
        (df.t / np.sqrt(flux_scale)).to_numpy(),
        dtype = torch.float32
    )

    source_age_tensor = torch.tensor(
        (df.source_age).to_numpy(),
        dtype = torch.float32
    )

    return torch.cat(
        (x_tensor.unsqueeze(0), 
         y_tensor.unsqueeze(0), 
         s_tensor.unsqueeze(0), 
         xflux_tensor.unsqueeze(0), 
         yflux_tensor.unsqueeze(0),
         velxerr_tensor.unsqueeze(0),
         velyerr_tensor.unsqueeze(0),
         vel_x_tensor.unsqueeze(0),
         vel_y_tensor.unsqueeze(0),
         t_tensor.unsqueeze(0),
         source_age_tensor.unsqueeze(0),
         torch.sqrt(torch.tensor(flux_scale)).repeat(1, source_age_tensor.shape[0])),
        dim = 0
    )
