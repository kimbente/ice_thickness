## Instructions for processing the real data

Preface: To reproduce the real data experiments, you can also just run the scripts on the provided, much smaller, preprocessed data. We include the following downloading and preprocessing pipeline for full **reproducibility** 

- In `download_bedmap123.py` replace `path_to_bedmap_data_folder` with your own local path. Run the python script with `python download_bedmap123.py` from the terminal. This will automatically download, unzip, and organise all bedmap data files. This script works on the os operating system. If you have trouble with this script or you are not on os, also see this [BAS resource from the Geophyscis Book by the UK Polar Centre](https://antarctica.github.io/PDC_GeophysicsBook/BEDMAP/Downloading_the_Bedmap_data.html) for useful information.
- <p style="color:red;"><strong>WARNING:</strong> This script downloads 11 GB of data!</p>
    - Bedmap1: 0.157 GB
    - Bedmap2: 3.2 GB
    - Bedmap3: 6.8 GB
- The script directly downloads all standardised .csv files from the Bedmap1, Bedmap2 and Bedmap3 collections from the [UK Polar Data Centre](https://www.bas.ac.uk/data/uk-pdc/). The list of .csv files are visible [on this Bristish Antarctic Survey (BAS) webpage](https://www.bas.ac.uk/project/bedmap/#data).
- Also check out this [Github repository](https://github.com/kimbente/bedmap) for some additional analysis of Bedmap123 data.

Created data structure in your `path_to_bedmap_data_folder`
- bedmap_raw_data/  
    - bedmap1_raw_data/ 
        - BEDMAP1_1966-2000_AIR_BM1.csv  
    - bedmap2_raw_data  
        - 
    - bedmap3_raw_data
        - BEDMAP3 - Ice thickness, bed and surface elevation for Antarctica - standardised data points/AWI_2013_GEA-IV_AIR_BM3.csv  
        - BEDMAP3 - Ice thickness, bed and surface elevation for Antarctica - standardised data points/AWI_2014_Recovery-Glacier_AIR_BM3.csv
        - [...]

## Intructions for preprocessing the real data
- In `preprocess_bedmap123.py`, specify your preference about which variable you care about by setting bool_remove_rows_without_ice_thickness and/or bool_remove_rows_without_bed_elevation. Also make sure you set `path_to_bedmap_data_folder` to the same path you used for the download scrip. We set `bool_remove_rows_without_ice_thickness = True` because we will be using ice thickness measurements.
    - For only `bool_remove_rows_without_ice_thickness = True` the resulting data set contains ~ 82 M points (i.e. rows.) and is 9.5 GB large.
    - For only `bool_remove_rows_without_bed_elevation = True` the resulting data set contains ~ 67 M points (i.e. rows.)
- The script combines all csv files into a standardised pd Dataframe and performs some cleaning and preprocessing steps.
- Number of csv files to combine: 151 
    - Number of bedmap1 csv files: 1
    - Number of bedmap2 csv files: 66
    - Number of bedmap3 csv files: 84
- The subset for the 300 x 300 km Byrd glacier catchments is more managable in size and only contains 750 k data points and is only 0.085 GB.

## The environment has been generated with 

``conda env export --prefix /home/kim/ice_thickness/.conda > environment.yml``



