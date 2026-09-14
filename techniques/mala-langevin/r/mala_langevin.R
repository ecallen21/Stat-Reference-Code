# Metropolis-adjusted Langevin (Roberts-Tweedie 1996)
# R: `mcmc::mcmc.metrop` (adaptive Langevin variants),
#    `LaplacesDemon::LaplacesDemon(Algorithm='MALA')`,
#    custom loop with autograd is common.
# Python: `blackjax.mala` (JAX), `pyro.infer.mcmc.MALA`,
#         from-scratch
#
# library(LaplacesDemon)
# fit <- LaplacesDemon(Model, ..., Algorithm = "MALA",
#                      Specs = list(A = 1e6, delta = 1))
