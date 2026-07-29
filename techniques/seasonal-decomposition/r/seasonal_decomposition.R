# Seasonal-trend decomposition (Reference §13.24, §13.47, §13.54)
# Base R via stats::decompose (classical) + stats::stl + seasonal::seas (X-13).
# Run with:  Rscript seasonal_decomposition.R

if (sys.nframe() == 0) {
  set.seed(23); n <- 120; m <- 12
  t <- 1:n
  y <- ts(10 + 0.05 * t + 2 * sin(2 * pi * t / m) + rnorm(n, 0, 0.4),
          frequency = m)
  cat("=== Classical decomposition ===\n"); print(decompose(y, type = "additive"))
  cat("\n=== STL decomposition ===\n"); print(stl(y, s.window = "periodic"))
  if (requireNamespace("seasonal", quietly = TRUE)) {
    cat("\n=== X-13ARIMA-SEATS via seasonal::seas ===\n")
    print(seasonal::seas(y))
  }
}
