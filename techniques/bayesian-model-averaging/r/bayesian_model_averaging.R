# Bayesian Model Averaging (Reference §14.26)
# R via BMA::bicreg or BAS::bas.lm.
# Run with:  Rscript bayesian_model_averaging.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 100; p <- 5
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1.5, -1, 0, 0.6, 0)
  y <- as.numeric(X %*% beta_true + rnorm(n))
  if (requireNamespace("BMA", quietly = TRUE)) {
    cat("=== BMA::bicreg (BIC-approximated BMA over all 2^p subsets) ===\n")
    print(summary(BMA::bicreg(X, y)))
  } else if (requireNamespace("BAS", quietly = TRUE)) {
    cat("=== BAS::bas.lm (marginal-likelihood BMA) ===\n")
    print(summary(BAS::bas.lm(y ~ ., data = data.frame(y = y, X))))
  }
}
