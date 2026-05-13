# Weighted Least Squares (Reference §5.10)
# From-scratch base R plus stats::lm(... weights = ...).
# Run with:  Rscript weighted_least_squares.R
#
# Inputs used below:
#   X       : n x p design matrix (first column = intercept)
#   y       : response vector
#   weights : positive weights of length n; typically 1 / Var(eps_i)

wls_fit_scratch <- function(X, y, weights) {
  X <- as.matrix(X); y <- as.numeric(y); w <- as.numeric(weights)
  stopifnot(all(w > 0))
  sw <- sqrt(w); Xw <- sw * X; yw <- sw * y
  beta <- as.vector(solve(crossprod(Xw), crossprod(Xw, yw)))
  yhat <- as.vector(X %*% beta); e <- y - yhat
  n <- nrow(X); p <- ncol(X)
  wrss <- sum(w * e^2); sigma2 <- wrss / (n - p)
  var_beta <- sigma2 * solve(crossprod(X * sqrt(w)))
  se <- sqrt(diag(var_beta))
  list(beta = beta, se = se, t_stats = beta / se,
       p_values = 2 * pt(-abs(beta / se), n - p),
       sigma_hat = sqrt(sigma2), df_residual = n - p,
       weighted_rss = wrss, residuals = e, fitted = yhat)
}

iteratively_reweighted_LS_scratch <- function(X, y, max_iter = 10, tol = 1e-6) {
  X <- as.matrix(X); n <- nrow(X); p <- ncol(X); w <- rep(1, n)
  beta_prev <- rep(0, p)
  it <- 0
  for (it in seq_len(max_iter)) {
    res <- wls_fit_scratch(X, y, w)
    if (it > 1 && max(abs(res$beta - beta_prev)) < tol) break
    beta_prev <- res$beta
    e <- res$residuals
    log_e2 <- log(pmax(e^2, 1e-12))
    gamma <- as.vector(solve(crossprod(X), crossprod(X, log_e2)))
    var_hat <- exp(as.vector(X %*% gamma))
    w <- 1 / pmax(var_hat, 1e-12)
  }
  res$iterations <- it; res
}

# Library: stats::lm(formula, weights = ...)
library_demo <- function(x, y, weights) {
  fit <- lm(y ~ x, weights = weights)
  cat("summary(lm with weights):\n"); print(summary(fit))
}

if (sys.nframe() == 0) {
  set.seed(6); n <- 200
  x <- runif(n, 1, 10)
  y <- 1 + 2 * x + rnorm(n, 0, 0.5 * x)        # heteroscedastic
  X <- cbind(1, x)
  cat("=== OLS (ignores heteroscedasticity) ===\n")
  print(wls_fit_scratch(X, y, rep(1, n))[c("beta", "se")])
  cat("\n=== WLS with KNOWN weights w_i = 1 / x_i^2 ===\n")
  print(wls_fit_scratch(X, y, 1 / x^2)[c("beta", "se")])
  cat("\n=== Iteratively reweighted ===\n")
  res_iter <- iteratively_reweighted_LS_scratch(X, y)
  cat("  beta =", res_iter$beta, "  SE =", res_iter$se, "  iter =", res_iter$iterations, "\n")
  cat("\n--- library ---\n"); library_demo(x, y, 1 / x^2)
}
