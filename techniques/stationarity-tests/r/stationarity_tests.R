# Stationarity tests: ADF, KPSS, Phillips-Perron (Reference §13.2, §13.8, §13.53)
# R via tseries::adf.test, tseries::kpss.test, tseries::pp.test.
# Run with:  Rscript stationarity_tests.R

if (sys.nframe() == 0) {
  set.seed(11); n <- 300
  rw <- cumsum(rnorm(n, 0.1, 1))                             # random walk
  ar <- numeric(n); ar[1] <- rnorm(1)
  for (t in 2:n) ar[t] <- 0.5 * ar[t - 1] + rnorm(1)          # stationary AR(1)
  if (requireNamespace("tseries", quietly = TRUE)) {
    cat("=== Random walk ===\n")
    print(tseries::adf.test(rw)); print(tseries::kpss.test(rw))
    cat("\n=== Differenced random walk ===\n")
    print(tseries::adf.test(diff(rw))); print(tseries::kpss.test(diff(rw)))
    cat("\n=== Stationary AR(1) ===\n")
    print(tseries::adf.test(ar)); print(tseries::kpss.test(ar))
    print(tseries::pp.test(rw))
  }
}
