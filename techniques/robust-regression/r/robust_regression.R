# Robust regression: Huber M-estimator (Reference §5.11)
# From-scratch base R plus MASS::rlm / robustbase::lmrob.
# Run with:  Rscript robust_regression.R
#
# Inputs used below:
#   X   : design matrix (first column = intercept)
#   y   : response vector
#   k   : Huber tuning constant (1.345 = ~95% efficiency at the normal)

mad_scale <- function(x) 1.4826 * median(abs(x - median(x)))

huber_regression_scratch <- function(X, y, k = 1.345, max_iter = 100, tol = 1e-7) {
  X <- as.matrix(X); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  beta <- as.vector(solve(crossprod(X), crossprod(X, y)))   # OLS start
  it <- 0
  for (it in seq_len(max_iter)) {
    r <- y - as.vector(X %*% beta)
    sigma <- mad_scale(r); if (sigma == 0) sigma <- 1
    u <- r / sigma
    w <- ifelse(abs(u) <= k, 1, k / pmax(abs(u), 1e-15))
    sw <- sqrt(w); Xw <- sw * X; yw <- sw * y
    beta_new <- as.vector(solve(crossprod(Xw), crossprod(Xw, yw)))
    if (max(abs(beta_new - beta)) < tol * max(1, max(abs(beta)))) {
      beta <- beta_new; break
    }
    beta <- beta_new
  }
  r <- y - as.vector(X %*% beta); sigma <- mad_scale(r)
  if (sigma == 0) sigma <- 1
  var_beta <- sigma^2 * solve(crossprod(X))
  se <- sqrt(diag(var_beta))
  list(beta = beta, se = se, sigma_hat = sigma, iterations = it,
       residuals = r, weights = w)
}

# Library: MASS::rlm (Huber default), robustbase::lmrob (MM-estimator),
#          robustbase::ltsReg (LTS), L1pack::lad (least-absolute-deviations)
library_demo <- function(x, y) {
  if (requireNamespace("MASS", quietly = TRUE)) {
    cat("MASS::rlm (Huber):\n"); print(summary(MASS::rlm(y ~ x)))
  } else cat("(install 'MASS' for rlm)\n")
  if (requireNamespace("robustbase", quietly = TRUE)) {
    cat("\nrobustbase::lmrob (MM-estimator):\n")
    print(summary(robustbase::lmrob(y ~ x)))
  }
}

if (sys.nframe() == 0) {
  set.seed(7); n <- 60
  x <- runif(n, 0, 10)
  y <- 1 + 2 * x + rnorm(n)
  y[1] <- 50; y[6] <- -40                # contamination
  X <- cbind(1, x)
  cat("=== OLS (contaminated) ===\n")
  print(as.vector(solve(crossprod(X), crossprod(X, y))))
  cat("\n=== Huber M-regression ===\n")
  print(huber_regression_scratch(X, y)[c("beta", "se", "iterations", "sigma_hat")])
  cat("\n--- library ---\n"); library_demo(x, y)
}
