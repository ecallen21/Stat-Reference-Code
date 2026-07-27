# Hotelling's T^2 test (Reference §9.1, §9.28)
# From-scratch base R + ICSNP::HotellingsT2 as library cross-check.
# Run with:  Rscript hotellings_t2.R
#
# Inputs:
#   X, X1, X2 : n x p numeric matrices
#   mu0       : length-p null-hypothesis mean vector (one-sample)

one_sample_t2 <- function(X, mu0) {
  X <- as.matrix(X); n <- nrow(X); p <- ncol(X)
  xbar <- colMeans(X)
  S <- cov(X)
  diff <- xbar - mu0
  T2 <- n * as.numeric(t(diff) %*% solve(S, diff))
  F_stat <- ((n - p) / (p * (n - 1))) * T2
  list(T_squared = T2, F = F_stat, df1 = p, df2 = n - p,
       p_value = pf(F_stat, p, n - p, lower.tail = FALSE),
       n = n, p_dim = p, mean_vector = xbar)
}

two_sample_t2 <- function(X1, X2) {
  X1 <- as.matrix(X1); X2 <- as.matrix(X2)
  n1 <- nrow(X1); n2 <- nrow(X2); p <- ncol(X1)
  m1 <- colMeans(X1); m2 <- colMeans(X2)
  S_pool <- ((n1 - 1) * cov(X1) + (n2 - 1) * cov(X2)) / (n1 + n2 - 2)
  diff <- m1 - m2
  T2 <- (n1 * n2 / (n1 + n2)) * as.numeric(t(diff) %*% solve(S_pool, diff))
  df2 <- n1 + n2 - p - 1
  F_stat <- (df2 / (p * (n1 + n2 - 2))) * T2
  list(T_squared = T2, F = F_stat, df1 = p, df2 = df2,
       p_value = pf(F_stat, p, df2, lower.tail = FALSE),
       n1 = n1, n2 = n2, p_dim = p, mean_diff = diff)
}

if (sys.nframe() == 0) {
  set.seed(11)
  mu_true <- c(1, 2, 3)
  Sigma <- matrix(c(1, 0.3, 0.2, 0.3, 1, 0.4, 0.2, 0.4, 1), 3, 3)
  X <- MASS::mvrnorm(60, mu_true, Sigma)
  Y <- MASS::mvrnorm(55, mu_true + c(0.5, 0, -0.3), Sigma)
  cat("=== One-sample vs true mean ===\n"); print(one_sample_t2(X, mu_true))
  cat("\n=== One-sample vs zero ===\n"); print(one_sample_t2(X, c(0, 0, 0)))
  cat("\n=== Two-sample ===\n"); print(two_sample_t2(X, Y))
  if (requireNamespace("ICSNP", quietly = TRUE)) {
    cat("\n--- library: ICSNP::HotellingsT2 ---\n")
    print(ICSNP::HotellingsT2(X, Y))
  }
}
