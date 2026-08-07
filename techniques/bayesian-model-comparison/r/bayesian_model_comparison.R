# Bayesian model comparison (Reference §14.20, §14.21, §14.22)
# R via loo::waic + loo::loo on pointwise log-likelihood matrices.
# Run with:  Rscript bayesian_model_comparison.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 200
  x <- rnorm(n); y <- 1 + 0.8 * x + rnorm(n)

  # Draw from analytic posteriors for two nested Bayesian LR models
  bayes_lr_draws <- function(X, y, n_draws = 1500) {
    p <- ncol(X)
    Vn <- solve(t(X) %*% X + diag(1e-4, p))
    mn <- as.numeric(Vn %*% t(X) %*% y)
    resid <- y - X %*% mn
    a_n <- n / 2 + 1; b_n <- 0.5 * sum(resid^2) + 1
    sig2 <- 1 / rgamma(n_draws, a_n, rate = b_n)
    betas <- t(sapply(sig2, function(s) MASS::mvrnorm(1, mn, s * Vn)))
    list(beta = betas, sig2 = sig2)
  }

  per_obs_ll <- function(X, y, betas, sig2) {
    ll <- matrix(0, length(sig2), length(y))
    for (s in seq_along(sig2)) {
      mu <- as.numeric(X %*% betas[s, ])
      ll[s, ] <- dnorm(y, mu, sqrt(sig2[s]), log = TRUE)
    }
    ll
  }

  X1 <- cbind(1, x); X2 <- matrix(1, n, 1)
  d1 <- bayes_lr_draws(X1, y); d2 <- bayes_lr_draws(X2, y)
  ll1 <- per_obs_ll(X1, y, d1$beta, d1$sig2)
  ll2 <- per_obs_ll(X2, y, d2$beta, d2$sig2)

  if (requireNamespace("loo", quietly = TRUE)) {
    cat("=== loo::waic ===\n")
    w1 <- loo::waic(ll1); w2 <- loo::waic(ll2)
    print(w1); print(w2)
    cat("=== loo::loo ===\n")
    l1 <- loo::loo(ll1); l2 <- loo::loo(ll2)
    print(loo::loo_compare(l1, l2))
  }
}
