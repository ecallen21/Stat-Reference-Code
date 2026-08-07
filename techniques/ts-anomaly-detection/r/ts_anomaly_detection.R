# Time series anomaly detection (Reference §13.29)
# R via pracma::hampel and forecast::tsoutliers.
# Run with:  Rscript ts_anomaly_detection.R

if (sys.nframe() == 0) {
  set.seed(0); T_ <- 300
  t <- 0:(T_ - 1)
  y <- 5 + 0.05 * t + 2 * sin(2 * pi * t / 30) + rnorm(T_, 0, 0.5)
  anom_idx <- c(50, 100, 175, 220, 280)
  y[anom_idx] <- y[anom_idx] + sample(c(-1, 1), 5, replace = TRUE) * 5
  if (requireNamespace("pracma", quietly = TRUE)) {
    cat("=== pracma::hampel ===\n")
    h <- pracma::hampel(y, k = 7, t0 = 3)
    detected <- h$ind
    cat("  detected:", detected, "\n")
    cat(sprintf("  precision = %.3f, recall = %.3f\n",
                length(intersect(detected, anom_idx)) / length(detected),
                length(intersect(detected, anom_idx)) / length(anom_idx)))
  }
  if (requireNamespace("forecast", quietly = TRUE)) {
    cat("\n=== forecast::tsoutliers (ARIMA residual approach) ===\n")
    tsr <- forecast::tsoutliers(ts(y, frequency = 30))
    cat("  index:", tsr$index, "\n")
  }
}
