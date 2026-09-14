# Lanczos iteration (Lanczos 1950)
# R: `RSpectra::eigs_sym(A, k, which='LA')` uses ARPACK's
#    implicitly-restarted Lanczos.
# Python: `scipy.sparse.linalg.eigsh` (ARPACK),
#         `torch.lobpcg`, from-scratch
#
# library(RSpectra)
# A <- crossprod(matrix(rnorm(400 * 400), 400))  # symmetric
# eigs_sym(A, k = 5, which = "LA")
