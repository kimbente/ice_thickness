# Experiments:

## Simulated experiments on divergence-free ice flux

3 examples
- funnel (show ice flux of outlet glaicer which is similar)
- merge
- Bifurication (morraine avoidance)
- curve (curving of ice flux my subglacial mountains for example)
- ridge 

Funnel: 
![alt text](image-2.png)

Curve:

### Baselines
- Non-phycial GPs (i.e. kriging)
- PINNs?! what training data? How do we comapre against this?

### Metrics
- divergence (to measure physical consistency)
- LML (propabilistc meassure)
- RMSE (error)

## Real data (Byrd glacier, Antarctica)

- Hold out measurements
    - because of spatial correlations hold-out in chess board scheme
- model ice flux
- depth-averaged mass balance 
- Show posterior samples for resulting ice flux and ice thickness distribution

### Comparisons
- compare ice flux divergence against BedMachine, and BedMap (existing maps)

### Ablations
- compare different covariance functions

# Method "Hard-constrained mass converving interpolation of ice thickness"

Based on this paper https://papers.nips.cc/paper_files/paper/2017/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf

- Potential issue with swirls: higher lengthscale may help
- 
