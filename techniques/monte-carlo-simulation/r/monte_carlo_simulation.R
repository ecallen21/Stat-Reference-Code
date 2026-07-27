# Monte Carlo simulation for power and CI coverage (Reference §10.9)
# From-scratch base R + stats::power.t.test for a cross-check.
# Run with:  Rscript monte_carlo_simulation.R

power_simulation <- function(sample_fn, test_fn, n_sim = 2000, alpha = 0.05, seed = 0) {
  set.seed(seed)
  reject <- replicate(n_sim, test_fn(sample_fn()) < alpha)
  power_hat <- mean(reject)
  list(power_hat = power_hat,
       MC_SE = sqrt(power_hat * (1 - power_hat) / n_sim),
       n_sim = n_sim, alpha = alpha)
}

coverage_simulation <- function(sample_fn, ci_fn, true_param, n_sim = 2000, seed = 0) {
  set.seed(seed)
  cov_widths <- t(replicate(n_sim, {
    ds <- sample_fn(); ci <- ci_fn(ds)
    c(cov = as.integer(ci[1] <= true_param && true_param <= ci[2]),
      w = ci[2] - ci[1])
  }))
  cov_hat <- mean(cov_widths[, "cov"])
  list(coverage_hat = cov_hat,
       MC_SE = sqrt(cov_hat * (1 - cov_hat) / n_sim),
       mean_CI_width = mean(cov_widths[, "w"]),
       n_sim = n_sim, true_param = true_param)
}

if (sys.nframe() == 0) {
  sample_ts <- function() list(x1 = rnorm(30), x2 = rnorm(30, 0.5))
  test_ts   <- function(ds) t.test(ds$x1, ds$x2, var.equal = FALSE)$p.value

  cat("=== Power: two-sample t at delta=0.5, n=30/gp ===\n")
  print(power_simulation(sample_ts, test_ts))
  cat("stats::power.t.test theoretical:\n")
  print(power.t.test(n = 30, delta = 0.5, sd = 1, alternative = "two.sided"))

  sample_sn <- function() rnorm(10, 5, 2)
  ci_mean   <- function(ds) {
    m <- mean(ds); s <- sd(ds); n <- length(ds)
    h <- qt(0.975, n - 1) * s / sqrt(n); c(m - h, m + h)
  }
  cat("\n=== Coverage: t 95% CI for mean, n=10 ===\n")
  print(coverage_simulation(sample_sn, ci_mean, true_param = 5))
}
