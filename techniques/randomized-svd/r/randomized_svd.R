# Randomized SVD (Halko-Martinsson-Tropp 2011)
# R: `rsvd` package — rsvd::rsvd(A, k=25, p=10, q=2)
# Python: `sklearn.utils.extmath.randomized_svd`, from-scratch
#
# library(rsvd)
# A <- matrix(rnorm(500 * 300), 500, 300)
# fit <- rsvd(A, k = 25, p = 10, q = 2)
# fit$d  # top-25 sigmas
