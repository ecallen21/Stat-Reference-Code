# Preconditioned Conjugate Gradient (Hestenes-Stiefel 1952;
# Concus-Golub-O'Leary 1976)
# R: `Matrix::solve(Sparse, ...)` (falls back to iterative),
#    `Rlinsolve::lsolve.pcg`, `pracma::conjugate_gradient`
# Python: `scipy.sparse.linalg.cg(preconditioned)`,
#         `scipy.sparse.linalg.LinearOperator` for M,
#         `pyamg` (algebraic multigrid preconditioners),
#         from-scratch
#
# library(Rlinsolve)
# A <- crossprod(matrix(rnorm(300*300), 300))
# b <- rnorm(300)
# fit <- lsolve.pcg(A, b, xinit = rep(0, 300),
#                   preconditioner = diag(diag(A)))
