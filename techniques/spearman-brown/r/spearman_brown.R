# Spearman-Brown prophecy + split-half (Reference §22.4)
# R via psych::splitHalf.
# Run with:  Rscript spearman_brown.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 300; K <- 10
  X <- rnorm(n) %*% t(rep(0.7, K)) + matrix(rnorm(n * K, 0, 0.5), n, K)
  if (requireNamespace("psych", quietly = TRUE)) {
    cat("=== psych::splitHalf ===\n")
    print(psych::splitHalf(X))
  }
}
