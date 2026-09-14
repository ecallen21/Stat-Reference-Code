# Chebyshev polynomial approximation (Chebyshev 1854)
# R: `chebpol::chebappxf`, `pracma::chebPoly`,
#    `pracma::chebApprox`
# Python: `numpy.polynomial.chebyshev.Chebyshev.fit`,
#         `scipy.interpolate.CubicSpline` (spline alternative),
#         `chebpy` (matlab-style chebfun), from-scratch
#
# library(chebpol)
# f <- function(x) 1 / (1 + 25 * x^2)
# ch <- ipol(f, dims = 1, intervals = c(-1, 1),
#            k = 40, method = "chebyshev")
