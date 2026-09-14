# Gauss-Legendre quadrature (Gauss 1814)
# R: `statmod::gauss.quad(n, kind='legendre')`,
#    `pracma::gaussLegendre`, `stats::integrate` (adaptive
#    Gauss-Kronrod via QUADPACK)
# Python: `numpy.polynomial.legendre.leggauss(n)`,
#         `scipy.integrate.fixed_quad`, `scipy.special.roots_legendre`,
#         from-scratch
#
# library(statmod)
# gq <- gauss.quad(16, kind = "legendre")
# f_shift <- function(x) exp(-x^2)
# sum(gq$weights * f_shift(gq$nodes))
