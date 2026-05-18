# Poisson regression (Reference §7.12, §7.43)
# From-scratch IRLS plus stats::glm(family = poisson).
# Run with:  Rscript poisson_regression.R

poisson_fit_scratch <- function(X, y, offset = NULL, max_iter = 50, tol = 1e-8) {
  X <- as.matrix(X); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  if (is.null(offset)) offset <- rep(0, n)
  eta <- log(pmax(y, 0.5)) - offset
  beta <- as.vector(solve(crossprod(X), crossprod(X, eta)))
  ll_prev <- -Inf; converged <- FALSE
  for (it in seq_len(max_iter)) {
    eta <- as.vector(X %*% beta)
    mu <- exp(pmin(pmax(eta + offset, -500), 500))
    w <- pmax(mu, 1e-12); z <- eta + (y - mu) / w
    sw <- sqrt(w); Xw <- sw * X; zw <- sw * z
    beta_new <- as.vector(solve(crossprod(Xw), crossprod(Xw, zw)))
    ll <- sum(y * log(pmax(mu, 1e-15)) - mu)
    if (abs(ll - ll_prev) < tol) { converged <- TRUE; break }
    ll_prev <- ll; beta <- beta_new
  }
  eta <- as.vector(X %*% beta); mu <- exp(eta + offset); w <- pmax(mu, 1e-12)
  var_beta <- solve(crossprod(X * sqrt(w))); se <- sqrt(diag(var_beta))
  z_stats <- beta / se
  pearson_chi2 <- sum((y - mu)^2 / pmax(mu, 1e-12))
  list(beta = beta, se = se, z = z_stats,
       p_values = 2 * pnorm(-abs(z_stats)),
       irr = exp(beta),
       irr_ci_lower = exp(beta - qnorm(0.975) * se),
       irr_ci_upper = exp(beta + qnorm(0.975) * se),
       pearson_chi_square = pearson_chi2,
       pearson_dispersion = pearson_chi2 / (n - p),
       deviance = 2 * sum(ifelse(y > 0, y * log(y / pmax(mu, 1e-12)), 0) - (y - mu)),
       iterations = it, converged = converged)
}

if (sys.nframe() == 0) {
  set.seed(4); n <- 300
  x1 <- rnorm(n); x2 <- rnorm(n); exposure <- runif(n, 0.5, 3.0)
  mu <- exposure * exp(0.5 + 0.6 * x1 - 0.3 * x2)
  y <- rpois(n, mu); X <- cbind(1, x1, x2)
  print(poisson_fit_scratch(X, y, offset = log(exposure)))
  cat("\n--- library ---\n")
  print(summary(glm(y ~ x1 + x2 + offset(log(exposure)), family = poisson)))
}
