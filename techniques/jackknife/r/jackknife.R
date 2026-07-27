# Jackknife SE + bias correction (Reference §10.6, §10.17)
# From-scratch base R + bootstrap::jackknife as library cross-check.
# Run with:  Rscript jackknife.R

jackknife_1d <- function(x, statistic) {
  n <- length(x); theta_hat <- statistic(x)
  J <- sapply(seq_len(n), function(i) statistic(x[-i]))
  Jbar <- mean(J)
  SE <- sqrt((n - 1) / n * sum((J - Jbar)^2))
  bias <- (n - 1) * (Jbar - theta_hat)
  list(theta_hat = theta_hat, jackknife_mean = Jbar,
       SE_jackknife = SE, bias_estimate = bias,
       theta_bias_corrected = theta_hat - bias,
       n = n)
}

jackknife_after_bootstrap <- function(x, statistic, n_boot = 2000, seed = 0) {
  set.seed(seed); n <- length(x)
  samples <- matrix(sample.int(n, n_boot * n, replace = TRUE), n_boot, n)
  theta_star <- apply(samples, 1, function(idx) statistic(x[idx]))
  se_full <- sd(theta_star)
  influence <- sapply(seq_len(n), function(i) {
    keep <- !apply(samples == i, 1, any)
    if (sum(keep) > 1) sd(theta_star[keep]) - se_full else NA
  })
  list(SE_full_bootstrap = se_full, influence = influence,
       top_5_influential = order(-abs(influence))[1:5], n = n, n_boot = n_boot)
}

if (sys.nframe() == 0) {
  set.seed(31); x <- rgamma(50, shape = 2, scale = 1)
  cat("=== Jackknife mean ===\n"); print(jackknife_1d(x, mean))
  cat("\n=== Jackknife biased-var (bias correction should recover unbiased var) ===\n")
  print(jackknife_1d(x, function(z) mean((z - mean(z))^2)))    # /n, biased
  cat("Compare unbiased var:", var(x), "\n")
  cat("\n=== Jackknife-after-bootstrap for mean ===\n")
  print(jackknife_after_bootstrap(x, mean))
  if (requireNamespace("bootstrap", quietly = TRUE)) {
    cat("\n--- library: bootstrap::jackknife ---\n")
    print(bootstrap::jackknife(x, mean))
  }
}
