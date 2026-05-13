# Multiple linear regression (Reference §5.2)
# From-scratch base R plus stats::lm. Run with:  Rscript multiple_linear_regression.R
#
# Inputs used below:
#   X  : numeric matrix, n rows x p cols (first column = 1s for the intercept)
#   y  : response vector of length n
#   names : optional column labels

fit_scratch <- function(X, y, names = NULL) {
  X <- as.matrix(X); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  if (is.null(names)) names <- paste0("x", seq_len(p) - 1L)
  beta <- as.vector(solve(crossprod(X), crossprod(X, y)))    # (X'X)^-1 X'y
  yhat <- as.vector(X %*% beta); resid <- y - yhat
  rss <- sum(resid^2); df_r <- n - p
  sigma2 <- rss / df_r; sigma <- sqrt(sigma2)
  XtX_inv <- solve(crossprod(X))
  var_beta <- sigma2 * XtX_inv
  se_beta <- sqrt(diag(var_beta))
  t_stats <- beta / se_beta
  p_vals  <- 2 * pt(-abs(t_stats), df_r)
  tc <- qt(0.975, df_r)
  ci_lower <- beta - tc * se_beta; ci_upper <- beta + tc * se_beta
  tss <- sum((y - mean(y))^2)
  r2 <- 1 - rss / tss
  adj_r2 <- 1 - (1 - r2) * (n - 1) / (n - p)
  has_int <- isTRUE(all.equal(X[, 1], rep(1, n)))
  F_overall <- if (has_int && p > 1) (r2 / (p - 1)) / ((1 - r2) / (n - p)) else NA
  p_F <- if (!is.na(F_overall)) pf(F_overall, p - 1, df_r, lower.tail = FALSE) else NA
  list(n = n, p = p, df_residual = df_r, names = names,
       beta = beta, se_beta = se_beta, t_stats = t_stats, p_values = p_vals,
       ci_lower = ci_lower, ci_upper = ci_upper,
       rss = rss, sigma_hat = sigma, r_squared = r2, adj_r_squared = adj_r2,
       F_overall = F_overall, p_F_overall = p_F,
       residuals = resid, fitted = yhat, vcov = var_beta)
}

predict_scratch <- function(fit_obj, X_new, conf = 0.95, kind = "confidence") {
  X_new <- as.matrix(X_new); V <- fit_obj$vcov
  yhat <- as.vector(X_new %*% fit_obj$beta)
  var_mean <- vapply(seq_len(nrow(X_new)),
                     function(i) as.numeric(t(X_new[i, ]) %*% V %*% X_new[i, ]),
                     numeric(1))
  var <- if (kind == "prediction") var_mean + fit_obj$sigma_hat^2 else var_mean
  se <- sqrt(var); tc <- qt(0.5 + conf / 2, fit_obj$df_residual)
  list(yhat = yhat, se = se, lower = yhat - tc * se,
       upper = yhat + tc * se, kind = kind)
}

# Library: stats::lm + summary, predict(... interval = "...")
library_demo <- function(df) {
  fit <- lm(y ~ x1 + x2 + x3, data = df)
  cat("summary(lm):\n"); print(summary(fit))
  cat("\nconfint:\n"); print(confint(fit))
}

if (sys.nframe() == 0) {
  set.seed(0); n <- 80
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
  y <- 1 + 2 * x1 - 1.5 * x2 + 0 * x3 + rnorm(n, 0, 1)
  X <- cbind(1, x1, x2, x3)
  res <- fit_scratch(X, y, names = c("intercept", "x1", "x2", "x3"))
  cat("=== Coefficient table ===\n")
  for (i in seq_along(res$beta))
    cat(sprintf("  %-10s %10.4f  SE %10.4f  t %8.3f  p %10.4g  [%.3f, %.3f]\n",
                res$names[i], res$beta[i], res$se_beta[i], res$t_stats[i],
                res$p_values[i], res$ci_lower[i], res$ci_upper[i]))
  cat(sprintf("\n  sigma = %.4f   df_resid = %d\n", res$sigma_hat, res$df_residual))
  cat(sprintf("  R^2 = %.4f   adj R^2 = %.4f\n", res$r_squared, res$adj_r_squared))
  cat(sprintf("  F(%d, %d) = %.3f   p = %.4g\n",
              res$p - 1, res$df_residual, res$F_overall, res$p_F_overall))
  cat("\n--- library ---\n"); library_demo(data.frame(y = y, x1 = x1, x2 = x2, x3 = x3))
}
