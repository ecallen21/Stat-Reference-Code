# Gaussian Process Regression (Reference §14.32)
# R via kernlab::gausspr, DiceKriging::km, or laGP::aGP.
# Run with:  Rscript gaussian_process_regression.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 20
  X <- matrix(runif(n, -3, 3), n, 1)
  y <- sin(1.5 * X[, 1]) + rnorm(n, 0, 0.15)
  X_star <- matrix(seq(-4, 4, length.out = 5), 5, 1)
  if (requireNamespace("DiceKriging", quietly = TRUE)) {
    cat("=== DiceKriging::km ===\n")
    km <- DiceKriging::km(design = data.frame(X), response = y,
                          nugget = 0.02, control = list(trace = FALSE))
    p <- DiceKriging::predict(km, newdata = data.frame(X = X_star), type = "SK")
    cat(sprintf("  predictions: %s\n", paste(round(p$mean, 3), collapse = " ")))
  } else if (requireNamespace("kernlab", quietly = TRUE)) {
    cat("=== kernlab::gausspr ===\n")
    fit <- kernlab::gausspr(X, y, kernel = "rbfdot", kpar = list(sigma = 0.5))
    print(predict(fit, X_star))
  }
}
