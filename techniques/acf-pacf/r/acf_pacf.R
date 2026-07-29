# ACF/PACF + Ljung-Box + CCF + Mann-Kendall (Reference §13.1, §13.9, §13.42, §13.48)
# Base R via stats::acf, stats::pacf, stats::Box.test, stats::ccf + Kendall::MannKendall.
# Run with:  Rscript acf_pacf.R

if (sys.nframe() == 0) {
  set.seed(3); n <- 200
  x <- numeric(n); x[1] <- rnorm(1)
  for (t in 2:n) x[t] <- 0.7 * x[t - 1] + rnorm(1)
  x <- x + 0.03 * seq_along(x)
  cat("=== ACF (first 6) ===\n"); print(acf(x, lag.max = 5, plot = FALSE)$acf[, , 1])
  cat("\n=== PACF (first 5) ===\n"); print(pacf(x, lag.max = 5, plot = FALSE)$acf[, , 1])
  cat("\n=== Ljung-Box ===\n")
  for (h in c(5, 10, 20)) print(Box.test(x, lag = h, type = "Ljung-Box"))
  y <- c(rep(NA, 2), head(x, n - 2)) + rnorm(n, 0, 0.5)
  cat("\n=== CCF x vs y (y = lag-2 x + noise) ===\n"); print(ccf(x, y, lag.max = 5, plot = FALSE)$acf[, , 1])
  cat("\n=== Mann-Kendall ===\n")
  if (requireNamespace("Kendall", quietly = TRUE)) print(Kendall::MannKendall(x))
}
