# Regression Discontinuity Design (Reference §15.9)
# R via rdrobust::rdrobust (Calonico-Cattaneo-Titiunik).
# Run with:  Rscript regression_discontinuity.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 500
  r <- runif(n, -2, 2); T <- as.integer(r >= 0)
  Y <- 1 + 0.5 * r + 2 * T + rnorm(n, 0, 0.6)
  if (requireNamespace("rdrobust", quietly = TRUE)) {
    cat("=== rdrobust::rdrobust (sharp) ===\n")
    print(rdrobust::rdrobust(y = Y, x = r, c = 0))
  }
}
