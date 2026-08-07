# Bayesian linear regression (Reference §14.10, §14.11)
# Base R Normal-InverseGamma closed-form posterior; canonical library
# alternatives are rstanarm::stan_glm or brms::brm.
# Run with:  Rscript bayesian_linear_regression.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 100; p <- 4
  X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
  beta_true <- c(1, 2, -1.5, 0.5)
  y <- as.numeric(X %*% beta_true + rnorm(n))

  # g-prior (Zellner): V0 = g (X'X)^-1
  g <- 100
  V0_inv <- (1 / g) * (t(X) %*% X)
  Vn_inv <- V0_inv + t(X) %*% X
  Vn <- solve(Vn_inv)
  mn <- Vn %*% (t(X) %*% y)
  a_n <- 0.001 + n / 2
  b_n <- 0.001 + 0.5 * (sum(y^2) - t(mn) %*% Vn_inv %*% mn)
  se <- sqrt(diag((b_n / (a_n - 1))[1] * Vn))
  df <- 2 * a_n
  cat("=== Bayesian LR with Zellner g-prior (g = 100) ===\n")
  for (i in 1:p) {
    cat(sprintf("  beta_%d: mean = %.3f, SE = %.3f, 95%% CrI = (%.3f, %.3f)  true = %g\n",
                i - 1, mn[i], se[i],
                mn[i] - qt(0.975, df) * se[i], mn[i] + qt(0.975, df) * se[i],
                beta_true[i]))
  }
}
