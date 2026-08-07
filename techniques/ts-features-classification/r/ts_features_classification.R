# Time series features + classification (Reference §13.39, §13.41)
# R via tsfeatures::tsfeatures and dtwclust for DTW-based classification.
# Run with:  Rscript ts_features_classification.R

if (sys.nframe() == 0) {
  set.seed(0)
  make_series <- function(cls, T_ = 100) {
    t <- seq(0, 4 * pi, length.out = T_)
    if (cls == 0) sin(t) + rnorm(T_, 0, 0.2)
    else if (cls == 1) cos(t) + rnorm(T_, 0, 0.2)
    else cumsum(rnorm(T_, 0, 0.3))
  }
  n_per <- 30
  X <- lapply(rep(0:2, each = n_per), make_series)
  y <- rep(0:2, each = n_per)

  if (requireNamespace("tsfeatures", quietly = TRUE)) {
    cat("=== tsfeatures on first 5 series ===\n")
    print(tsfeatures::tsfeatures(X[1:5]))
  } else {
    cat("tsfeatures not installed; skip.\n")
  }
}
