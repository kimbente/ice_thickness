# Ice thickness

Directly model ice thickness from Bedmap3 data points. Byrd glacier.

![alt text](image.png)

# ToDo:
- domain-informed noise level
- std to count ratio: standard error

## Why do we model ice thickness and not the bed elevation?
- Hypothesis: Ice thickness distribution is smoother in space and thus easier to model with the choosen methods (GPs/kriging/kernel methods are naturally better at modelling smooth distributions.)
- Test this hypothesis with
    - Look at std
    - Fit simple GP model and interpret

## Why do we go directly from measurements to high-resolution bed topography models
- Because uncertainty quantification is much better this way.

## Research plans
- Investigate per grid error and number of data points
- Visualise number of data points
- Investigate roughness: Roughness ML module for post-processing needed? 
- Integrate MC ideas in kernel: physical consistency

## Considerations
- Check that geoid/ellipsoid reference is consistent. BedMachine uses ellipsoid.
- Check that height is consitent too. BedMachine fird-corrects to attain ice-equivalent values.
- Remove flight line that is dubious. ()

# Data

- Bedmap123 data preprocess and subsetted for this region around Byrd glacier.
    - **surface_altitude**: Surface elevation or altitude (referenced to WGS84) in meters
    - **land_ice_thickness**: Ice thickness in meters
    - **bedrock_altitude** Bed elevation or altitude (referenced to WGS84) in meters
- Bedmachine v3
    - In BedMachine Antarctica, all heights are referenced to mean sea level (using the geoid EIGEN-6C4). To
convert the heights to heights referenced to the WGS84 ellipsoid, simply add the geoid height
    - Ice equivalent units.

Corrections of BedMachine:
- Bed, thickness, and surface elevation need to be corrected to ellipsoid. 
- Ice thickness also needs to be firn corrected.

# Scaleable inference

https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/index.html 

# Gradient observation

https://docs.gpytorch.ai/en/v1.11/examples/08_Advanced_Usage/Simple_GP_Regression_Derivative_Information_1d.html