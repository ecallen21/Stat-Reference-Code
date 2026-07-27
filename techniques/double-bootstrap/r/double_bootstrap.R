# Double (iterated) bootstrap for CI calibration (Reference §10.11)
# From-scratch base R.
# Run with:  Rscript double_bootstrap.R

double_bootstrap <- function(x, statistic, n_boot_outer = 200,
                              n_boot_inner = 200, conf = 0.95, seed = 0) {
  set.seed(seed); n <- length(x); theta_hat <- statistic(x)
  outer_theta <- numeric(n_boot_outer); covered <- numeric(n_boot_outer)
  alpha0 <- 1 - conf
  for (b in seq_len(n_boot_outer)) {
    x_star <- x[sample.int(n, n, replace = TRUE)]
    outer_theta[b] <- statistic(x_star)
    inner <- replicate(n_boot_inner, statistic(x_star[sample.int(n, n, replace = TRUE)]))
    q <- quantile(inner, c(alpha0 / 2, 1 - alpha0 / 2))
    covered[b] <- as.integer(q[[1]] <= theta_hat && theta_hat <= q[[2]])
  }
  emp_cov <- mean(covered)
  alpha_emp <- max(1e-6, 1 - emp_cov)
  alpha_cal <- max(1e-6, min(0.5, alpha0^2 / alpha_emp))
  qp <- quantile(outer_theta, c(alpha0 / 2, 1 - alpha0 / 2))
  qc <- quantile(outer_theta, c(alpha_cal / 2, 1 - alpha_cal / 2))
  list(theta_hat = theta_hat,
       empirical_inner_coverage = emp_cov, nominal_level = conf,
       CI_plain = c(lower = qp[[1]], upper = qp[[2]]),
       CI_calibrated = c(lower = qc[[1]], upper = qc[[2]]),
       alpha_calibrated = alpha_cal,
       n_boot_outer = n_boot_outer, n_boot_inner = n_boot_inner)
}

if (sys.nframe() == 0) {
  set.seed(53); x <- rexp(50, rate = 0.5)
  cat("=== Double bootstrap for MEDIAN ===\n")
  print(double_bootstrap(x, median))
}
