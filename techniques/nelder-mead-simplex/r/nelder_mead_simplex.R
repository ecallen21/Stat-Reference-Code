# Nelder-Mead simplex (Nelder-Mead 1965)
# R: `optim(par, fn, method='Nelder-Mead')`,
#    `dfoptim::nmk` (bounded variant)
# Python: `scipy.optimize.minimize(method='Nelder-Mead')`,
#         from-scratch
#
# rosen <- function(x) sum(100 * (x[-1] - x[-length(x)]^2)^2 +
#                          (1 - x[-length(x)])^2)
# optim(c(-1.2, 1.0), rosen, method = "Nelder-Mead")
