# BCa bootstrap CI + comparison of bootstrap CIs (Reference §10.3, §10.14)
# From-scratch base R + boot::boot.ci as library cross-check.
# Run with:  Rscript bca_bootstrap.R

bootstrap_reps <- function(x, statistic, n_boot, seed) {
  set.seed(seed); n <- length(x)
  replicate(n_boot, statistic(x[sample.int(n, n, replace = TRUE)]))
}

jackknife_reps <- function(x, statistic) {
  n <- length(x); sapply(seq_len(n), function(i) statistic(x[-i]))
}

bca_ci <- function(x, statistic, n_boot = 2000, conf = 0.95, seed = 0) {
  n <- length(x); theta_hat <- statistic(x)
  theta_star <- bootstrap_reps(x, statistic, n_boot, seed)
  p <- mean(theta_star < theta_hat); p <- min(max(p, 1e-12), 1 - 1e-12)
  z0 <- qnorm(p)
  J <- jackknife_reps(x, statistic); Jbar <- mean(J)
  num <- sum((Jbar - J)^3); den <- 6 * sum((Jbar - J)^2)^1.5
  a <- if (den > 0) num / den else 0
  alpha <- 1 - conf
  adjust <- function(zq) pnorm(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
  a1 <- adjust(qnorm(alpha / 2)); a2 <- adjust(qnorm(1 - alpha / 2))
  q <- quantile(theta_star, c(a1, a2))
  list(theta_hat = theta_hat, z0 = z0, a = a,
       adjusted_percentiles = c(lower = a1, upper = a2),
       CI_BCa = c(lower = q[[1]], upper = q[[2]]),
       n_boot = n_boot, conf = conf)
}

compare_ci_methods <- function(x, statistic, n_boot = 2000, conf = 0.95, seed = 0) {
  n <- length(x); theta_hat <- statistic(x)
  theta_star <- bootstrap_reps(x, statistic, n_boot, seed)
  alpha <- 1 - conf; q <- quantile(theta_star, c(alpha / 2, 1 - alpha / 2))
  z <- qnorm(1 - alpha / 2); se <- sd(theta_star)
  # BCa
  p <- mean(theta_star < theta_hat); p <- min(max(p, 1e-12), 1 - 1e-12); z0 <- qnorm(p)
  J <- jackknife_reps(x, statistic); Jbar <- mean(J)
  num <- sum((Jbar - J)^3); den <- 6 * sum((Jbar - J)^2)^1.5
  a <- if (den > 0) num / den else 0
  adjust <- function(zq) pnorm(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
  qb <- quantile(theta_star, c(adjust(qnorm(alpha / 2)), adjust(qnorm(1 - alpha / 2))))
  list(
    CI_percentile = c(lower = q[[1]], upper = q[[2]]),
    CI_basic      = c(lower = 2 * theta_hat - q[[2]], upper = 2 * theta_hat - q[[1]]),
    CI_normal     = c(lower = theta_hat - z * se, upper = theta_hat + z * se),
    CI_BCa        = c(lower = qb[[1]], upper = qb[[2]]),
    theta_hat = theta_hat, SE_bootstrap = se, z0 = z0, a = a
  )
}

if (sys.nframe() == 0) {
  set.seed(11); x <- rexp(80, rate = 0.5)
  cat("=== BCa CI for median ===\n"); print(bca_ci(x, median))
  cat("\n=== Compare CI methods ===\n"); print(compare_ci_methods(x, median))
  if (requireNamespace("boot", quietly = TRUE)) {
    cat("\n--- library: boot::boot.ci ---\n")
    b <- boot::boot(x, function(d, i) median(d[i]), R = 2000)
    print(boot::boot.ci(b, type = c("perc", "basic", "bca", "norm")))
  }
}
