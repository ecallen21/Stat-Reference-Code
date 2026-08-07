# Approximate Bayesian Computation (Reference §14.27)
# R via the abc package.
# Run with:  Rscript abc_approximate_bayesian.R

if (sys.nframe() == 0) {
  set.seed(0); n_obs <- 50; true_mu <- 3
  y_obs <- rnorm(n_obs, true_mu, 1)
  sum_obs <- c(mean(y_obs), sd(y_obs))
  N <- 20000
  mus <- runif(N, -5, 10)
  sim <- t(sapply(mus, function(m) {
    y <- rnorm(n_obs, m, 1); c(mean(y), sd(y))
  }))
  if (requireNamespace("abc", quietly = TRUE)) {
    cat("=== abc::abc (rejection + local-linear regression adjustment) ===\n")
    fit <- abc::abc(target = sum_obs, param = mus, sumstat = sim,
                    tol = 0.01, method = "loclinear")
    cat(sprintf("  posterior mean (adjusted): %.3f  95%% CI (%.3f, %.3f)\n",
                mean(fit$adj.values), quantile(fit$adj.values, 0.025),
                quantile(fit$adj.values, 0.975)))
  } else {
    cat("Manual rejection ABC:\n")
    d <- sqrt(rowSums(sweep(sim, 2, sum_obs)^2))
    keep <- order(d)[1:200]
    cat(sprintf("  posterior mean: %.3f  95%% CI (%.3f, %.3f)\n",
                mean(mus[keep]), quantile(mus[keep], 0.025),
                quantile(mus[keep], 0.975)))
  }
}
