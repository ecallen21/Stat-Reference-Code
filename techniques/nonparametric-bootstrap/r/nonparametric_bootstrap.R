# Nonparametric bootstrap (Reference §10.1)
# From-scratch base R + boot::boot as library cross-check.
# Run with:  Rscript nonparametric_bootstrap.R

bootstrap_1d <- function(x, statistic, n_boot = 2000, conf = 0.95, seed = 0) {
  set.seed(seed); n <- length(x); theta_hat <- statistic(x)
  theta_star <- replicate(n_boot, statistic(x[sample.int(n, n, replace = TRUE)]))
  alpha <- 1 - conf
  q <- quantile(theta_star, c(alpha / 2, 1 - alpha / 2))
  z <- qnorm(1 - alpha / 2); se <- sd(theta_star)
  list(theta_hat = theta_hat, bootstrap_SE = se,
       CI_percentile = c(lower = q[[1]], upper = q[[2]]),
       CI_basic      = c(lower = 2 * theta_hat - q[[2]], upper = 2 * theta_hat - q[[1]]),
       CI_normal     = c(lower = theta_hat - z * se, upper = theta_hat + z * se),
       n_boot = n_boot, n = n, conf = conf)
}

if (sys.nframe() == 0) {
  set.seed(4); x <- rexp(100, rate = 0.5)      # mean = 2, skewed
  cat("=== Bootstrap median (n=100, B=2000) ===\n")
  print(bootstrap_1d(x, median))
  if (requireNamespace("boot", quietly = TRUE)) {
    cat("\n--- library: boot::boot ---\n")
    b <- boot::boot(x, function(d, i) median(d[i]), R = 2000)
    print(boot::boot.ci(b, type = c("perc", "basic", "norm")))
  }
}
