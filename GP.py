############
### NLML ###
############

# In this version we use the precomputed alpha and L matrices to compute the NLML to save compute
# Negative Log Marginal Likelihood (NLML) is the objective function that we want to minimize
# equivalent to maximising the lml
# for HYPERPARAMETER OPTIMIZATION

# Rasmussen and Williams 2006 Algorithm 2.1, line 8, page 37
# Note: Ramussen does not call it negative but it is negative
# see https://gregorygundersen.com/blog/2019/09/12/practical-gp-regression/ for derivations

# TERM 1: Model-data fit
# 0.5 * y^T * alpha
# where y^T is a row vector of shape [1, n] and alpha is a column vector of shape [n, 1] so that the product is a scalar

# TERM 2: Complexity
# sum(log(L_ii)))
# where L is the lower triangular matrix of the Cholesky decomposition of the covariance matrix K
# we sum the diagonal elements of L after taking the log

# TERM 3: Normalisation term
# n/2 * log(2*pi)
# where n is the number of data points
# this is a constant term