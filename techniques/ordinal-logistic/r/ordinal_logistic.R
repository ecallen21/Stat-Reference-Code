# Ordinal logistic regression (Proportional Odds) (Reference §7.2)
# Library: MASS::polr is the canonical implementation. From-scratch via optim.
# Run with:  Rscript ordinal_logistic.R
#
# Inputs:  X (n x p, no intercept), y (ordered integer 1..K)

build_params <- function(theta, K, p) {
  alpha <- numeric(K - 1); alpha[1] <- theta[1]
  if (K > 2) for (k in 2:(K - 1)) alpha[k] <- alpha[k - 1] + exp(theta[k])
  beta <- theta[K:(K + p - 1)]
  list(alpha = alpha, beta = beta)
}

neg_log_lik <- function(theta, X, y, K) {
  pars <- build_params(theta, K, ncol(X))
  alpha <- pars$alpha; beta <- pars$beta
  eta_extra <- -as.vector(X %*% beta)
  ll <- 0
  for (i in seq_len(nrow(X))) {
    cum_prev <- 0
    for (k in seq_len(K)) {
      cum <- if (k < K) 1 / (1 + exp(-(alpha[k] + eta_extra[i]))) else 1
      pk <- cum - cum_prev
      if (y[i] == k) ll <- ll + log(max(pk, 1e-15))
      cum_prev <- cum
    }
  }
  -ll
}

fit_scratch <- function(X, y) {
  X <- as.matrix(X); y <- as.integer(y); K <- max(y); p <- ncol(X)
  cum_props <- pmax(pmin(sapply(seq_len(K - 1), function(k) mean(y <= k)),
                         1 - 1e-6), 1e-6)
  alpha0 <- log(cum_props / (1 - cum_props))
  theta0 <- c(alpha0[1], log(diff(alpha0)), rep(0, p))
  res <- optim(theta0, neg_log_lik, X = X, y = y, K = K, method = "BFGS",
               hessian = TRUE)
  pars <- build_params(res$par, K, p)
  H_inv <- solve(res$hessian); se_beta <- sqrt(diag(H_inv))[K:(K + p - 1)]
  z <- pars$beta / se_beta; p_vals <- 2 * pnorm(-abs(z))
  zc <- qnorm(0.975)
  list(alpha = pars$alpha, beta = pars$beta, se_beta = se_beta,
       z = z, p_values = p_vals, odds_ratio = exp(pars$beta),
       or_ci_lower = exp(pars$beta - zc * se_beta),
       or_ci_upper = exp(pars$beta + zc * se_beta),
       K = K, log_likelihood = -res$value, converged = (res$convergence == 0))
}

library_demo <- function(X, y) {
  if (requireNamespace("MASS", quietly = TRUE)) {
    df <- data.frame(y = ordered(y), X)
    print(summary(MASS::polr(y ~ ., data = df, Hess = TRUE)))
  } else cat("(install 'MASS' for polr)\n")
}

if (sys.nframe() == 0) {
  set.seed(1); n <- 400
  x1 <- rnorm(n); x2 <- rnorm(n)
  eta <- 1.0 * x1 - 0.7 * x2
  noise <- -log(1 / runif(n, 1e-9, 1 - 1e-9) - 1)
  y_star <- eta + noise
  y <- cut(y_star, c(-Inf, -1, 0.5, 2, Inf), labels = FALSE)
  X <- cbind(x1, x2)
  print(fit_scratch(X, y))
  cat("\n--- library ---\n"); library_demo(X, y)
}
