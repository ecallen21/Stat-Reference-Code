# Distance correlation (Reference §4.8)
# From-scratch base R plus energy::dcor / energy::dcorT.
# Run with:  Rscript distance_correlation.R
#
# Inputs used below:
#   x, y    : numeric vectors of equal length (also works for matrices in 'energy')
#   n_perm  : permutation replications for the independence test

double_centered <- function(M) {
  rm <- rowMeans(M); cm <- colMeans(M); gm <- mean(M)
  M - rm - rep(cm, each = nrow(M)) + gm
}

distance_correlation_scratch <- function(x, y) {
  stopifnot(length(x) == length(y))
  n <- length(x)
  a <- abs(outer(x, x, "-")); b <- abs(outer(y, y, "-"))
  A <- double_centered(a); B <- double_centered(b)
  dcov2 <- mean(A * B); dvarx2 <- mean(A * A); dvary2 <- mean(B * B)
  dcor <- if (dvarx2 > 0 && dvary2 > 0) sqrt(dcov2 / sqrt(dvarx2 * dvary2)) else 0
  list(dCor = dcor, dCov_sq = dcov2, dVar_sq_x = dvarx2, dVar_sq_y = dvary2,
       energy_statistic_nT = n * dcov2, n = n)
}

distance_correlation_test_scratch <- function(x, y, n_perm = 500, seed = 0) {
  set.seed(seed); obs <- distance_correlation_scratch(x, y)$dCor
  sim <- vapply(seq_len(n_perm),
                function(i) distance_correlation_scratch(x, sample(y))$dCor,
                numeric(1))
  list(dCor = obs, p_value = mean(sim >= obs), n_permutations = n_perm)
}

# Library: energy::dcor (point estimate), energy::dcor.test (permutation)
library_demo <- function(x, y) {
  if (requireNamespace("energy", quietly = TRUE)) {
    cat("energy::dcor:", energy::dcor(x, y), "\n")
    print(energy::dcor.test(x, y, R = 300))
  } else cat("(install 'energy' for distance correlation)\n")
}

if (sys.nframe() == 0) {
  set.seed(42)
  cat("=== Independent x, y ~ N(0, 1) ===\n")
  x <- rnorm(200); y <- rnorm(200)
  print(distance_correlation_test_scratch(x, y, n_perm = 300))
  cat("\n=== Quadratic: y = x^2 + noise (Pearson r ~ 0) ===\n")
  x <- seq(-3, 3, length.out = 200); y <- x^2 + rnorm(200, 0, 0.4)
  print(distance_correlation_scratch(x, y))
  print(distance_correlation_test_scratch(x, y, n_perm = 300))
  cat("\n--- library ---\n"); library_demo(x, y)
}
