# ARIMA (Reference §13.4, §13.5, §13.52)
# Base R via stats::arima + forecast::auto.arima for order selection.
# Run with:  Rscript arima.R

if (sys.nframe() == 0) {
  set.seed(13); n <- 300
  x <- as.numeric(arima.sim(model = list(ar = 0.6, ma = 0.4), n = n))
  cat("=== stats::arima(x, order = c(1, 0, 1)) ===\n")
  fit <- arima(x, order = c(1, 0, 1))
  print(fit)
  cat("\n=== Residual Ljung-Box ===\n")
  print(Box.test(residuals(fit), lag = 20, type = "Ljung-Box"))
  if (requireNamespace("forecast", quietly = TRUE)) {
    cat("\n=== forecast::auto.arima (BIC by default) ===\n")
    print(forecast::auto.arima(x))
  }
}
