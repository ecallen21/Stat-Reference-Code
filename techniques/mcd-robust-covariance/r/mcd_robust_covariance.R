# MCD - Minimum Covariance Determinant (Rousseeuw 1985)
# R: `robustbase::covMcd` (the Rousseeuw reference),
#    `rrcov::CovMcd`, `MASS::cov.mve` (older MVE)
# Python: `sklearn.covariance.MinCovDet`,
#         `sklearn.covariance.EllipticEnvelope`
#         (outlier detection wrapper), from-scratch FAST-MCD
#
# library(robustbase)
# fit <- covMcd(X, alpha = 0.75)
# fit$center; fit$cov
