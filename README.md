# Ice thickness

Directly model ice thickness from Bedmap3 data points. Byrd glacier.

![alt text](image.png)

# ToDo:
- domain-informed noise level

## Why do we model ice thickness and not the bed elevation?
- Hypothesis: Ice thickness distribution is smoother in space and thus easier to model with the choosen methods. 
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