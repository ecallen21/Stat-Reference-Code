# Block bootstrap for dependent data (Reference §10.4)
# From-scratch base R + boot::tsboot (block bootstrap in R's boot package).
# Run with:  Rscript block_bootstrap.R

moving_block_bootstrap <- function(x, statistic, block_length,
                                    n_boot = 2000, conf = 0.95, seed = 0) {
  set.seed(seed); n <- length(x); L <- block_length
  n_blocks <- ceiling(n / L); n_starts <- n - L + 1
  theta_hat <- statistic(x)
  theta_star <- replicate(n_boot, {
    starts <- sample.int(n_starts, n_blocks, replace = TRUE)
    xb <- unlist(lapply(starts, function(s) x[s:(s + L - 1)]))[1:n]
    statistic(xb)
  })
  alpha <- 1 - conf; q <- quantile(theta_star, c(alpha / 2, 1 - alpha / 2))
  list(theta_hat = theta_hat, bootstrap_SE = sd(theta_star),
       CI_percentile = c(lower = q[[1]], upper = q[[2]]),
       block_length = L, n_boot = n_boot, n = n)
}

circular_block_bootstrap <- function(x, statistic, block_length,
                                      n_boot = 2000, conf = 0.95, seed = 0) {
  set.seed(seed); n <- length(x); L <- block_length
  n_blocks <- ceiling(n / L); x_ext <- c(x, x[seq_len(L - 1)])
  theta_hat <- statistic(x)
  theta_star <- replicate(n_boot, {
    starts <- sample.int(n, n_blocks, replace = TRUE)
    xb <- unlist(lapply(starts, function(s) x_ext[s:(s + L - 1)]))[1:n]
    statistic(xb)
  })
  alpha <- 1 - conf; q <- quantile(theta_star, c(alpha / 2, 1 - alpha / 2))
  list(theta_hat = theta_hat, bootstrap_SE = sd(theta_star),
       CI_percentile = c(lower = q[[1]], upper = q[[2]]),
       block_length = L, n_boot = n_boot, n = n)
}

rule_of_thumb_L <- function(n) max(1, round(n^(1/3)))

if (sys.nframe() == 0) {
  set.seed(19); n <- 300; phi <- 0.7
  x <- numeric(n); x[1] <- rnorm(1)
  for (t in 2:n) x[t] <- phi * x[t - 1] + rnorm(1)
  L <- rule_of_thumb_L(n)
  cat("=== Moving-block, L =", L, "===\n")
  print(moving_block_bootstrap(x, mean, L))
  cat("\n=== Circular block ===\n")
  print(circular_block_bootstrap(x, mean, L))
  if (requireNamespace("boot", quietly = TRUE)) {
    cat("\n--- library: boot::tsboot ---\n")
    b <- boot::tsboot(x, mean, R = 2000, l = L, sim = "fixed")
    print(boot::boot.ci(b, type = "perc"))
  }
}
