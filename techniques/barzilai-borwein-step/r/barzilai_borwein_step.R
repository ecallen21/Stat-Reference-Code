# Barzilai-Borwein spectral step (Barzilai-Borwein 1988)
# R: `optimx::optimx(method='BB')` (uses BBoptim / spg),
#    `BB::BBoptim` (Ravi Varadhan)
# Python: `scipy.optimize.minimize` does not expose BB directly;
#         `pyproximal` proximal gradient with BB step,
#         from-scratch
#
# library(BB)
# fit <- BBoptim(par = rep(0, 50), fn = f, gr = grad,
#                control = list(maxit = 200))
