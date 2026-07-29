# Multivariate outlier detection (Reference §9.6, §9.7)
# R via stats::mahalanobis + robustbase::covMcd for the robust version.
# Run with:  Rscript mv_outlier_detection.R

if (sys.nframe() == 0) {
  set.seed(3); n <- 100; p <- 2
  X <- rbind(MASS::mvrnorm(n - 8, mu = c(0, 0), Sigma = diag(p)),
             MASS::mvrnorm(8,     mu = c(6, 6), Sigma = 0.3 * diag(p)))
  cutoff <- qchisq(0.975, p)

  cat("=== Classical Mahalanobis ===\n")
  D2 <- mahalanobis(X, center = colMeans(X), cov = cov(X))
  cat("  flagged:", which(D2 > cutoff), "\n")

  if (requireNamespace("robustbase", quietly = TRUE)) {
    cat("\n=== robustbase::covMcd (Fast-MCD) ===\n")
    fit <- robustbase::covMcd(X, alpha = 0.75)
    D2r <- mahalanobis(X, center = fit$center, cov = fit$cov)
    cat("  flagged:", which(D2r > cutoff), "\n")
  }
}
