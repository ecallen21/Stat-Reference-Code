# Gaussian Mixture Models via EM (Reference §9.12)
# Base R relies on mclust for the standard implementation.
# Run with:  Rscript gaussian_mixture_models.R

if (sys.nframe() == 0) {
  set.seed(71); n_per <- 150
  X1 <- MASS::mvrnorm(n_per, c(0, 0), matrix(c(1, 0.2, 0.2, 0.6), 2))
  X2 <- MASS::mvrnorm(n_per, c(5, 5), matrix(c(0.7, -0.3, -0.3, 1.2), 2))
  X3 <- MASS::mvrnorm(n_per, c(1, 3), matrix(c(0.5, 0, 0, 0.5), 2))
  X <- rbind(X1, X2, X3)
  if (requireNamespace("mclust", quietly = TRUE)) {
    fit <- mclust::Mclust(X, G = 1:6)
    cat("=== mclust::Mclust (G in 1..6) ===\n")
    print(summary(fit))
    cat("\nBIC by G:\n"); print(fit$BIC)
  } else {
    cat("mclust not installed; skipping.\n")
  }
}
