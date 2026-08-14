# Path analysis (Reference §19.4)
# R via lavaan::sem for the general case.
# Run with:  Rscript path_analysis.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 500
  W <- rnorm(n); M <- 0.6 * W + rnorm(n); Y <- 0.4 * M + 0.3 * W + rnorm(n)
  df <- data.frame(W = W, M = M, Y = Y)
  if (requireNamespace("lavaan", quietly = TRUE)) {
    cat("=== lavaan::sem for a small path model ===\n")
    fit <- lavaan::sem('M ~ W
                        Y ~ M + W', data = df)
    print(lavaan::parameterEstimates(fit))
  }
}
