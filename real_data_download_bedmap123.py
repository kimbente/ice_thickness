import os
import subprocess
import shutil

# WARNING: This script will create new directories and download large amounts of data
# CHANGE THIS AS NEEDED, specifiying your (full) local path to an empty data folder or a (not yet existing) folder that you want to download the data into. The following is an example:
path_to_bedmap_data_folder = "/home/kim/data/bedmap_raw_data_test"

# Subdirectories for different Bedmap versions
path_to_bedmap1_data_folder = os.path.join(path_to_bedmap_data_folder, "bedmap1_raw_data")
path_to_bedmap2_data_folder = os.path.join(path_to_bedmap_data_folder, "bedmap2_raw_data")
path_to_bedmap3_data_folder = os.path.join(path_to_bedmap_data_folder, "bedmap3_raw_data")

# Create all required directories
for folder in [
    path_to_bedmap_data_folder,
    path_to_bedmap1_data_folder,
    path_to_bedmap2_data_folder,
    path_to_bedmap3_data_folder
]:
    os.makedirs(folder, exist_ok = True)
    print(f"Created directory: {folder}")

# --- Bedmap1: Download CSV file ---
path_to_bedmap1_csv = os.path.join(path_to_bedmap1_data_folder, "BEDMAP1_1966-2000_AIR_BM1.csv")
bedmap1_url = (
    "https://ramadda.data.bas.ac.uk/repository/entry/get/"
    "BEDMAP1_1966-2000_AIR_BM1.csv?entryid=synth%3Af64815ec-4077-4432-9f55-0ce230f46029%3AL0JFRE1BUDFfMTk2Ni0yMDAwX0FJUl9CTTEuY3N2"
)

print(f"\nDownloading Bedmap1 CSV to {path_to_bedmap1_csv} ...")
subprocess.run(["wget", "-O", path_to_bedmap1_csv, bedmap1_url], check=True)
print("Bedmap1 download complete.")

# --- Bedmap2: Download and extract ZIP ---
bedmap2_zip_path = os.path.join(path_to_bedmap2_data_folder, "bedmap2.zip")
bedmap2_url = (
    "https://ramadda.data.bas.ac.uk/repository/entry/show?"
    "entryid=2fd95199-365e-4da1-ae26-3b6d48b3e6ac&output=zip.tree"
)

print(f"\nDownloading Bedmap2 ZIP to {bedmap2_zip_path} ...")
subprocess.run(["wget", "-O", bedmap2_zip_path, bedmap2_url], check=True)
print("Bedmap2 ZIP download complete.")

print(f"\nExtracting Bedmap2 ZIP into {path_to_bedmap2_data_folder} ...")
subprocess.run(["unzip", "-o", bedmap2_zip_path, "-d", path_to_bedmap2_data_folder], check=True)
print("Extraction complete.")

# Move files from nested folder to main bedmap2 folder
nested_folder = os.path.join(
    path_to_bedmap2_data_folder,
    "BEDMAP2 - Ice thickness, bed and surface elevation for Antarctica - standardised data points"
)

if os.path.isdir(nested_folder):
    for filename in os.listdir(nested_folder):
        src_path = os.path.join(nested_folder, filename)
        dst_path = os.path.join(path_to_bedmap2_data_folder, filename)
        shutil.move(src_path, dst_path)
        print(f"Moved: {filename}")
    os.rmdir(nested_folder)
    print("Removed nested folder.")
else:
    print("Warning: Expected nested folder not found.")


# --- Bedmap3: Download, unzip, move, cleanup ---
bedmap3_zip_path = os.path.join(path_to_bedmap3_data_folder, "bedmap3.zip")
bedmap3_url = (
    "https://ramadda.data.bas.ac.uk/repository/entry/show?"
    "entryid=91523ff9-d621-46b3-87f7-ffb6efcd1847&output=zip.tree"
)
print(f"\nDownloading Bedmap3 ZIP to {bedmap3_zip_path} ...")
subprocess.run(["wget", "-O", bedmap3_zip_path, bedmap3_url], check=True)
print("Bedmap3 ZIP download complete.")

print(f"\nExtracting Bedmap3 ZIP into {path_to_bedmap3_data_folder} ...")
subprocess.run(["unzip", "-o", bedmap3_zip_path, "-d", path_to_bedmap3_data_folder], check=True)

# Remove the ZIP file
os.remove(bedmap3_zip_path)
print("Removed Bedmap3 ZIP.")

# Move files from nested folder to main bedmap3 folder
bedmap3_nested = os.path.join(
    path_to_bedmap3_data_folder,
    "BEDMAP3 - Ice thickness, bed and surface elevation for Antarctica - standardised data points"
)
if os.path.isdir(bedmap3_nested):
    for filename in os.listdir(bedmap3_nested):
        shutil.move(os.path.join(bedmap3_nested, filename), path_to_bedmap3_data_folder)
    os.rmdir(bedmap3_nested)
    print("Moved Bedmap3 files and cleaned up nested folder.")
else:
    print("Warning: Bedmap3 nested folder not found.")