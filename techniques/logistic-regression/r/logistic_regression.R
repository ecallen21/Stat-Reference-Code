# Binary logistic regression (Reference §7.1)
# From-scratch IRLS plus stats::glm(family = binomial).
# Run with:  Rscript logistic_regression.R
#
# Inputs:  X (design matrix incl. intercept), y (0/1 response)

logistic_fit_scratch <- function(X, y, max_iter = 50, tol = 1e-8) {
  X <- as.matrix(X); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  beta <- rep(0, p); ll_prev <- -Inf; converged <- FALSE; it <- 0
  for (it in seq_len(max_iter)) {
    eta <- as.vector(X %*% beta); pi <- 1 / (1 + exp(-pmin(pmax(eta, -500), 500)))
    w <- pmax(pi * (1 - pi), 1e-12)
    z <- eta + (y - pi) / w
    sw <- sqrt(w); Xw <- sw * X; zw <- sw * z
    beta_new <- as.vector(solve(crossprod(Xw), crossprod(Xw, zw)))
    ll <- sum(y * log(pmax(pi, 1e-15)) + (1 - y) * log(pmax(1 - pi, 1e-15)))
    if (abs(ll - ll_prev) < tol) { converged <- TRUE; break }
    ll_prev <- ll; beta <- beta_new
  }
  eta <- as.vector(X %*% beta); pi <- 1 / (1 + exp(-eta))
  w <- pmax(pi * (1 - pi), 1e-12)
  var_beta <- solve(crossprod(X * sqrt(w))); se <- sqrt(diag(var_beta))
  z_stats <- beta / se; p_vals <- 2 * pnorm(-abs(z_stats))
  or_  <- exp(beta); zc <- qnorm(0.975)
  or_lo <- exp(beta - zc * se); or_hi <- exp(beta + zc * se)
  p_bar <- mean(y)
  ll_null <- n * (p_bar * log(max(p_bar, 1e-15)) + (1 - p_bar) * log(max(1 - p_bar, 1e-15)))
  ll_full <- sum(y * log(pmax(pi, 1e-15)) + (1 - y) * log(pmax(1 - pi, 1e-15)))
  list(beta = beta, se = se, z = z_stats, p_values = p_vals,
       odds_ratio = or_, or_ci_lower = or_lo, or_ci_upper = or_hi,
       log_likelihood = ll_full, ll_null = ll_null,
       lr_chi_square = 2 * (ll_full - ll_null),
       lr_df = p - 1,
       lr_p_value = pchisq(2 * (ll_full - ll_null), p - 1, lower.tail = FALSE),
       null_deviance = -2 * ll_null, residual_deviance = -2 * ll_full,
       aic = 2 * p - 2 * ll_full,
       mcfadden_r_squared = 1 - ll_full / ll_null,
       accuracy_at_0.5 = mean((pi >= 0.5) == y),
       iterations = it, converged = converged)
}

library_demo <- function(X_no_const, y) {
  fit <- glm(y ~ X_no_const, family = binomial)
  cat("summary(glm):\n"); print(summary(fit))
  cat("\nodds ratios + 95% CI:\n"); print(exp(cbind(coef(fit), confint.default(fit))))
}

if (sys.nframe() == 0) {
  set.seed(0); n <- 200
  x1 <- rnorm(n); x2 <- rnorm(n)
  eta <- -0.5 + 1.0 * x1 - 0.7 * x2
  p <- 1 / (1 + exp(-eta)); y <- as.integer(runif(n) < p)
  X <- cbind(1, x1, x2)
  res <- logistic_fit_scratch(X, y)
  cat("=== Fit ===\n")
  for (i in seq_along(res$beta))
    cat(sprintf("  %-8s beta=%+.4f  SE=%.4f  z=%+.2f  p=%.4g  OR=%.4f [%.3f, %.3f]\n",
                c("intercept", "x1", "x2")[i], res$beta[i], res$se[i],
                res$z[i], res$p_values[i], res$odds_ratio[i],
                res$or_ci_lower[i], res$or_ci_upper[i]))
  cat(sprintf("\nLL=%.4f  AIC=%.4f  LR chi2(%d)=%.4f  p=%.4g  McFadden R^2=%.4f\n",
              res$log_likelihood, res$aic, res$lr_df,
              res$lr_chi_square, res$lr_p_value, res$mcfadden_r_squared))
  cat("\n--- library ---\n"); library_demo(cbind(x1, x2), y)
}
