# Exponential Smoothing family (Reference §13.3, §13.43, §13.56)
# Base R via stats::HoltWinters + forecast::ets + forecast::tbats.
# Run with:  Rscript exponential_smoothing.R

if (sys.nframe() == 0) {
  set.seed(19); n <- 120; m <- 12
  t <- 1:n
  y <- ts(10 + 0.05 * t + 2 * sin(2 * pi * t / m) + rnorm(n, 0, 0.5), frequency = m)
  cat("=== stats::HoltWinters (additive) ===\n"); print(HoltWinters(y))
  if (requireNamespace("forecast", quietly = TRUE)) {
    cat("\n=== forecast::ets ===\n"); print(forecast::ets(y))
    cat("\n=== forecast::tbats (multiple seasonalities support) ===\n")
    print(summary(forecast::tbats(y)))
  }
}
