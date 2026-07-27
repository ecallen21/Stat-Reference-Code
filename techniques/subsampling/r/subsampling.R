# Subsampling and m-out-of-n bootstrap (Reference §10.10, §10.15)
# From-scratch base R.
# Run with:  Rscript subsampling.R

subsampling_1d <- function(x, statistic, m, n_sub = 2000,
                            conf = 0.95, rate_pow = 0.5, seed = 0) {
  set.seed(seed); n <- length(x)
  if (m >= n) stop("m must be < n")
  theta_hat <- statistic(x)
  theta_sub <- replicate(n_sub, statistic(sample(x, m, replace = FALSE)))
  a_m <- m^rate_pow; a_n <- n^rate_pow
  T_sub <- a_m * (theta_sub - theta_hat)
  q <- quantile(T_sub, c((1 - conf) / 2, 1 - (1 - conf) / 2))
  list(theta_hat = theta_hat,
       subsample_SE_at_m = sd(theta_sub),
       CI_subsampling = c(lower = theta_hat - q[[2]] / a_n,
                          upper = theta_hat - q[[1]] / a_n),
       m = m, n = n, n_sub = n_sub)
}

m_out_of_n_bootstrap <- function(x, statistic, m, n_boot = 2000,
                                  conf = 0.95, seed = 0) {
  set.seed(seed); n <- length(x); theta_hat <- statistic(x)
  theta_star <- replicate(n_boot, statistic(sample(x, m, replace = TRUE)))
  q <- quantile(theta_star, c((1 - conf) / 2, 1 - (1 - conf) / 2))
  list(theta_hat = theta_hat, bootstrap_SE_at_m = sd(theta_star),
       CI_percentile = c(lower = q[[1]], upper = q[[2]]),
       m = m, n = n, n_boot = n_boot)
}

if (sys.nframe() == 0) {
  set.seed(43); x <- rexp(200, rate = 1)
  cat("=== Subsampling CI for MAX ===\n"); print(subsampling_1d(x, max, m = 50))
  cat("\n=== m-out-of-n bootstrap for MAX ===\n"); print(m_out_of_n_bootstrap(x, max, m = 50))
}
