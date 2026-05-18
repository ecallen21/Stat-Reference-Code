# Negative binomial regression (Reference §7.13)
# From-scratch joint BFGS plus MASS::glm.nb.
# Run with:  Rscript negative_binomial_regression.R

nb_neg_log_lik <- function(params, X, y, offset) {
  p <- ncol(X); beta <- params[seq_len(p)]; theta <- exp(params[p + 1])
  eta <- as.vector(X %*% beta) + offset
  mu <- exp(pmin(pmax(eta, -500), 500))
  -sum(lgamma(y + theta) - lgamma(theta) - lgamma(y + 1)
       + theta * log(theta / (theta + mu)) + y * log(mu / (theta + mu)))
}

nb_fit_scratch <- function(X, y, offset = NULL) {
  X <- as.matrix(X); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  if (is.null(offset)) offset <- rep(0, n)
  eta0 <- log(pmax(y, 0.5)) - offset
  beta0 <- as.vector(solve(crossprod(X), crossprod(X, eta0)))
  init <- c(beta0, 0)
  res <- optim(init, nb_neg_log_lik, X = X, y = y, offset = offset,
               method = "BFGS", hessian = TRUE,
               control = list(reltol = 1e-9))
  beta <- res$par[seq_len(p)]; theta <- exp(res$par[p + 1])
  se_all <- sqrt(diag(solve(res$hessian))); se_beta <- se_all[seq_len(p)]
  list(beta = beta, se = se_beta, z = beta / se_beta,
       p_values = 2 * pnorm(-abs(beta / se_beta)),
       irr = exp(beta),
       irr_ci_lower = exp(beta - qnorm(0.975) * se_beta),
       irr_ci_upper = exp(beta + qnorm(0.975) * se_beta),
       theta = theta, alpha = 1 / theta,
       log_likelihood = -res$value,
       converged = (res$convergence == 0))
}

if (sys.nframe() == 0) {
  set.seed(5); n <- 400
  x1 <- rnorm(n); x2 <- rnorm(n)
  mu_true <- exp(0.5 + 0.6 * x1 - 0.4 * x2); theta_true <- 2
  y <- rnbinom(n, size = theta_true, mu = mu_true)
  X <- cbind(1, x1, x2)
  print(nb_fit_scratch(X, y))
  cat("\n--- library ---\n")
  if (requireNamespace("MASS", quietly = TRUE))
    print(summary(MASS::glm.nb(y ~ x1 + x2)))
  else cat("(install 'MASS' for glm.nb)\n")
}
