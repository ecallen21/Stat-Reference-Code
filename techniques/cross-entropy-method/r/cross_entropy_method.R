# Cross-Entropy method (Rubinstein 1997)
# R: `CEoptim::CEoptim` (Kroese & Chan port);
#    `nloptr` NEWUOA / COBYLA (loosely related).
# Python: `CEM.py` in torchrl for policy search;
#         `pypi cma` for CMA-ES cousin; from-scratch
#
# library(CEoptim)
# fit <- CEoptim(rastrigin,
#                continuous = list(mean = rep(0, 5),
#                                  sd   = rep(4, 5)),
#                rho = 0.1, N = 500, iter = 100)
