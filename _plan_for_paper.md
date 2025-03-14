# Experiments:

## Simulated experiments on divergence-free ice flux

5 representative examples

simulate_convergence, simulate_merge, simulate_branching, simulate_deflection, simulate_ridge
- (1) Convergence
- (2) Merge (& curve, e.g. due to subglacial mountain)
- (3) Branching (opposite of converging in a way)
- (4) Deflection (i.e. bifurication, e.g. morraine avoidance)
- (5) Ridge (hard example, maybe tweek slighly as it is not the most "representative" example)

We can also show examples from real ice flux that resemble the simulations.

### Baselines

- Ours
    - 2d lengthscales
    - https://github.com/carji475/linearly-constrained-gaussian-processes/blob/master/simulation-example/simulation_study.m
- Non-phycial GPs (i.e. kriging)
- PINNs (soft-constrained)
    - https://www.thomasteisberg.com/projects/igarss2021/
    - https://github.com/thomasteisberg/igarss2021 
    - https://egusphere.copernicus.org/preprints/2024/egusphere-2024-1732/egusphere-2024-1732.pdf 
- Neural Conservation Laws (NCLs) i.e. Hard-Constrained Neural Networks (HCNNs)
    - https://arxiv.org/abs/2210.01741 

Ablations:
- mean function
- covariance functions

### Metrics

On test data
- divergence (to measure physical consistency)
- NLL (propabilistc meassure)
- RMSE
- MAE

## Real data (Byrd glacier, Antarctica)

- Hold out measurements
    - because of spatial correlations hold-out in chess board scheme
- model ice flux
- depth-averaged mass balance 
- Show posterior samples for resulting ice flux and ice thickness distribution

### Comparisons
- compare ice flux divergence against BedMachine, and BedMap (existing maps)

# Names for our method

-  "Hard-constrained mass converving interpolation of ice thickness"
- GPs underPINNed 

