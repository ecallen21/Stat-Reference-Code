# Probit regression (Reference §7.10)
# From-scratch IRLS plus stats::glm(family = binomial(link = "probit")).
# Run with:  Rscript probit_regression.R

probit_fit_scratch <- function(X, y, max_iter = 50, tol = 1e-8) {
  X <- as.matrix(X); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  beta <- rep(0, p); ll_prev <- -Inf; converged <- FALSE
  for (it in seq_len(max_iter)) {
    eta <- as.vector(X %*% beta)
    pi <- pmax(pmin(pnorm(eta), 1 - 1e-12), 1e-12)
    phi <- dnorm(eta)
    w <- pmax(phi^2 / (pi * (1 - pi)), 1e-12)
    z <- eta + (y - pi) / phi
    sw <- sqrt(w); Xw <- sw * X; zw <- sw * z
    beta_new <- as.vector(solve(crossprod(Xw), crossprod(Xw, zw)))
    ll <- sum(y * log(pi) + (1 - y) * log(1 - pi))
    if (abs(ll - ll_prev) < tol) { converged <- TRUE; break }
    ll_prev <- ll; beta <- beta_new
  }
  eta <- as.vector(X %*% beta)
  pi <- pmax(pmin(pnorm(eta), 1 - 1e-12), 1e-12); phi <- dnorm(eta)
  w <- phi^2 / (pi * (1 - pi))
  var_beta <- solve(crossprod(X * sqrt(w))); se <- sqrt(diag(var_beta))
  z_stats <- beta / se
  list(beta = beta, se = se, z = z_stats,
       p_values = 2 * pnorm(-abs(z_stats)),
       average_marginal_effect = mean(phi) * beta,
       log_likelihood = sum(y * log(pi) + (1 - y) * log(1 - pi)),
       converged = converged, iterations = it)
}

if (sys.nframe() == 0) {
  set.seed(3); n <- 300
  x1 <- rnorm(n); x2 <- rnorm(n)
  eta <- -0.3 + 0.8 * x1 - 0.5 * x2
  y <- as.integer(runif(n) < pnorm(eta))
  X <- cbind(1, x1, x2)
  print(probit_fit_scratch(X, y))
  cat("\n--- library ---\n")
  print(summary(glm(y ~ x1 + x2, family = binomial(link = "probit"))))
}
