import os
import pandas as pd
import pyproj

# NOTE: This script might take a while to run and requires sufficient memory

# Set preferences depending on which variable we are interested in. We will use ice thickness data for now, so we only exclude (i.e. drop) rows without ice thickness.
# NOTE: Of course data may be combined with REMA surface elevation measurements (however with temporal mismatch) to infer missing variables within the s (surface elevation) - t (ice thickness) - b (bed elevation) "triangle".

bool_remove_rows_without_ice_thickness = True
bool_remove_rows_without_bed_elevation = False

bool_save_byrd_catchment_crop = True

# CHANGE THIS TO YOUR PATH, same as in the download script
path_to_bedmap_data_folder = "/home/kim/data/bedmap_raw_data_test"

# paths to subfolders
path_to_bedmap1_data_folder = os.path.join(path_to_bedmap_data_folder, "bedmap1_raw_data")
path_to_bedmap2_data_folder = os.path.join(path_to_bedmap_data_folder, "bedmap2_raw_data")
path_to_bedmap3_data_folder = os.path.join(path_to_bedmap_data_folder, "bedmap3_raw_data")

# list all CSVs in the folder
list_of_bedmap1_csv_files = [f for f in os.listdir(path_to_bedmap1_data_folder) if f.endswith(".csv")]
list_of_bedmap2_csv_files = [f for f in os.listdir(path_to_bedmap2_data_folder) if f.endswith(".csv")]
list_of_bedmap3_csv_files = [f for f in os.listdir(path_to_bedmap3_data_folder) if f.endswith(".csv")]

# initialise DataFrame and column names
column_list = ["lon", "lat", "x", "y", "s", "t", "b", "b_inferred", "source"]
bedmap123_data = pd.DataFrame(columns = column_list)

# set up coordinate transformer once
lonlat_to_polarstereo = pyproj.Transformer.from_crs(
    crs_from = pyproj.CRS("epsg:4326"), # WGS84 (lon, lat)
    crs_to = pyproj.CRS("epsg:3031"), # Antarctic Polar Stereographic (x, y)
    always_xy = True
)

# lists 
paths_to_data_folders_all_versions = [path_to_bedmap1_data_folder, path_to_bedmap2_data_folder, path_to_bedmap3_data_folder]
list_of_all_versions = [list_of_bedmap1_csv_files, list_of_bedmap2_csv_files, list_of_bedmap3_csv_files]

# loop over bedmap versions
for v, (csv_list, folder_path) in enumerate(zip(list_of_all_versions, paths_to_data_folders_all_versions), start = 1):
    print(f"Processing Bedmap{v}...")
    print(f"Number of bedmap{v} csv files:", len(csv_list))

    # loop over csv files
    for i in csv_list:

        print("Processing:", i)
        # construct full file path
        file_path = os.path.join(folder_path, i)

        # Load CSV, skipping metadata header lines
        pd_data = pd.read_csv(file_path, skiprows = 18, low_memory = False)

        # Extract and rename required columns
        df = pd_data[[
            "longitude (degree_east)",
            "latitude (degree_north)",
            "surface_altitude (m)",
            "land_ice_thickness (m)",
            "bedrock_altitude (m)"
        ]].copy() # NOTE: Copy to avoid SettingWithCopyWarning

        # Rename columns to short names
        df.columns = ["lon", "lat", "s", "t", "b"]

        # Mark where bedrock elevation is inferred (as this is approximately true)
        df["b_inferred"] = False
        # Create a mask where bedrock elevation is missing (-9999) but surface and land ice thickness are provided
        infer_b_mask = (df['s'] != -9999) & (df['t'] != -9999) & (df['b'] == -9999)
        df.loc[infer_b_mask, 'b_inferred'] = True
        df.loc[infer_b_mask, 'b'] = df['s'] - df['t']

        # Drop rows still missing bed elevation
        # TODO: Change this if we are focussing on ice thickness and not bed elevation
        if bool_remove_rows_without_bed_elevation:
            dropped_rows_b = (df['b'] == -9999).sum()
            print(f"#rows dropped: {dropped_rows_b}")
            df = df[df['b'] != -9999]
        
        if bool_remove_rows_without_ice_thickness:
            dropped_rows_t = (df['t'] == -9999).sum()
            print(f"#rows dropped: {dropped_rows_t}")
            df = df[df['t'] != -9999]

        # Project coordinates 
        df["x"], df["y"] = lonlat_to_polarstereo.transform(df["lon"].values, df["lat"].values)

        # Add filename as source
        df["source"] = i

        # Ensure column order and append
        df = df[column_list]
        bedmap123_data = pd.concat([bedmap123_data, df], ignore_index = True)

    # End csv_list loop
    # Final check of shape
    print("Combined dataset shape:", bedmap123_data.shape)

# End version loop
print(f"Finished processing Bedmap{v}.")

# Save the combined DataFrame to a CSV file. This step takes a while, so be patient.
# NOTE: This will overwrite the file if it already exists
output_path = os.path.join(path_to_bedmap_data_folder, "bedmap123.csv")
bedmap123_data.to_csv(output_path, index = False)

print(f"Combined dataset saved to {output_path}.")

if bool_save_byrd_catchment_crop:
    # Hard-coded corners of 300 x 300 km Byrd catchment (crop). In the domain literature the domain is defined as an even larger region, however we are interested in the faster flowing areas closer to Byrd glacier drainage. 
    x_min = 350 * 1000
    x_max = 650 * 1000
    y_min = -1000 * 1000
    y_max = -700 * 1000

    # Subset the DataFrame to the Byrd catchment area
    bedmap123_data_byrd_catchment = bedmap123_data[
        (bedmap123_data["x"] >= x_min) & (bedmap123_data["x"] <= x_max) &
        (bedmap123_data["y"] >= y_min) & (bedmap123_data["y"] <= y_max)
    ]

    output_path_byrd = os.path.join(path_to_bedmap_data_folder, "bedmap123_byrd_catchment.csv")
    bedmap123_data_byrd_catchment.to_csv(output_path_byrd, index = False)
    print(f"Byrd catchment subset saved to {output_path_byrd}.")