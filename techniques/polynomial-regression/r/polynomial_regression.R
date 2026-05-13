# Polynomial regression (Reference §5.3)
# From-scratch base R plus stats::poly (raw = FALSE for orthogonal).
# Run with:  Rscript polynomial_regression.R
#
# Inputs used below:
#   x       : numeric predictor
#   y       : numeric response
#   degree  : polynomial degree
#   raw     : TRUE = [1, x, x^2, ...]; FALSE = orthonormal polynomials

poly_features_scratch <- function(x, degree, raw = TRUE) {
  P <- sapply(0:degree, function(k) x^k)
  if (raw) return(P)
  Q <- matrix(0, nrow = nrow(P), ncol = ncol(P))
  for (k in seq_len(ncol(P))) {
    v <- P[, k]
    if (k > 1) for (j in seq_len(k - 1)) v <- v - sum(Q[, j] * v) * Q[, j]
    norm <- sqrt(sum(v^2)); Q[, k] <- if (norm > 0) v / norm else v
  }
  Q
}

fit_polynomial_scratch <- function(x, y, degree, raw = TRUE) {
  X <- poly_features_scratch(x, degree, raw); n <- nrow(X); p <- ncol(X)
  beta <- as.vector(solve(crossprod(X), crossprod(X, y)))
  yhat <- as.vector(X %*% beta); e <- y - yhat
  rss <- sum(e^2); df_r <- n - p; sigma2 <- rss / df_r
  var_beta <- sigma2 * solve(crossprod(X)); se <- sqrt(diag(var_beta))
  t_stats <- beta / se; p_vals <- 2 * pt(-abs(t_stats), df_r)
  tss <- sum((y - mean(y))^2)
  list(degree = degree, raw = raw, beta = beta, se = se,
       t_stats = t_stats, p_values = p_vals,
       r_squared = 1 - rss / tss, sigma_hat = sqrt(sigma2), df_residual = df_r,
       fitted = yhat, residuals = e)
}

choose_degree_scratch <- function(x, y, max_degree = 6, alpha = 0.05) {
  results <- list(); chosen <- 1
  for (d in 1:max_degree) {
    fit <- fit_polynomial_scratch(x, y, d, raw = FALSE)
    results[[d]] <- list(degree = d, top_t = fit$t_stats[length(fit$t_stats)],
                         top_p = fit$p_values[length(fit$p_values)],
                         r_squared = fit$r_squared)
    if (fit$p_values[length(fit$p_values)] < alpha) chosen <- d
  }
  list(results = results, chosen_degree = chosen)
}

# Library: poly(x, degree, raw = ...) inside lm()
library_demo <- function(x, y, degree = 3) {
  cat("lm(y ~ poly(x, 3, raw = TRUE)):\n")
  print(summary(lm(y ~ poly(x, degree, raw = TRUE))))
  cat("\nlm(y ~ poly(x, 3)) (orthogonal by default):\n")
  print(summary(lm(y ~ poly(x, degree))))
}

if (sys.nframe() == 0) {
  set.seed(2); n <- 100
  x <- seq(-3, 3, length.out = n) + rnorm(n, 0, 0.1)
  y <- 1 + 0.5 * x + 0.8 * x^2 + rnorm(n, 0, 0.7)
  cat("=== Raw polynomial, degree 3 ===\n"); print(fit_polynomial_scratch(x, y, 3, TRUE))
  cat("\n=== Orthogonal polynomial, degree 3 ===\n"); print(fit_polynomial_scratch(x, y, 3, FALSE))
  cat("\n=== Choose degree (alpha = 0.05) ===\n")
  sel <- choose_degree_scratch(x, y, max_degree = 6)
  for (r in sel$results)
    cat(sprintf("  d = %d: top t = %+.3f  p = %.4g  R^2 = %.4f\n",
                r$degree, r$top_t, r$top_p, r$r_squared))
  cat(sprintf("  Chosen degree: %d\n", sel$chosen_degree))
  cat("\n--- library ---\n"); library_demo(x, y)
}
