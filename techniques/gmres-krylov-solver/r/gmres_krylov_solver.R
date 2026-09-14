# GMRES (Saad-Schultz 1986)
# R: `Rlinsolve::lsolve.gmres`, `pracma::gmres`,
#    `Matrix::solve` (falls back to iterative when sparse)
# Python: `scipy.sparse.linalg.gmres`,
#         `scipy.sparse.linalg.lgmres`, from-scratch
#
# library(Rlinsolve)
# A <- matrix(rnorm(200*200), 200) + 200*diag(200)
# b <- rnorm(200)
# fit <- lsolve.gmres(A, b, xinit = rep(0, 200), reorth = 10)
