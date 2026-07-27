# Permutation tests (Reference §10.7, §10.16)
# From-scratch base R + coin::oneway_test / coin::spearman_test as library cross-check.
# Run with:  Rscript permutation_tests.R

perm_two_sample <- function(x1, x2, statistic = NULL,
                             n_perm = 5000, alternative = "two.sided", seed = 0) {
  set.seed(seed); n1 <- length(x1); n2 <- length(x2)
  combined <- c(x1, x2); n <- n1 + n2
  if (is.null(statistic)) statistic <- function(a, b) mean(a) - mean(b)
  t_obs <- statistic(x1, x2)
  t_perm <- replicate(n_perm, {
    idx <- sample.int(n)
    statistic(combined[idx[1:n1]], combined[idx[(n1 + 1):n]])
  })
  extreme <- switch(alternative,
    "two.sided" = abs(t_perm) >= abs(t_obs),
    "greater"   = t_perm >= t_obs,
    "less"      = t_perm <= t_obs)
  p <- (1 + sum(extreme)) / (1 + n_perm)
  list(T_obs = t_obs, p_value = p, alternative = alternative,
       n_perm = n_perm, n1 = n1, n2 = n2)
}

perm_correlation <- function(x, y, n_perm = 5000, seed = 0) {
  set.seed(seed); r_obs <- cor(x, y)
  r_perm <- replicate(n_perm, cor(x, sample(y)))
  p <- (1 + sum(abs(r_perm) >= abs(r_obs))) / (1 + n_perm)
  list(r_obs = r_obs, p_value = p, n_perm = n_perm)
}

perm_regression_coef <- function(X, y, coef_index, n_perm = 5000, seed = 0) {
  set.seed(seed); X <- as.matrix(X)
  beta_obs <- lm.fit(X, y)$coefficients
  obs <- beta_obs[coef_index]
  coef_perm <- replicate(n_perm, lm.fit(X, sample(y))$coefficients[coef_index])
  p <- (1 + sum(abs(coef_perm) >= abs(obs))) / (1 + n_perm)
  list(coef_obs = obs, p_value = p, coef_index = coef_index, n_perm = n_perm)
}

if (sys.nframe() == 0) {
  set.seed(37)
  x1 <- rnorm(40); x2 <- rnorm(45, mean = 0.5)
  cat("=== Two-sample permutation ===\n"); print(perm_two_sample(x1, x2))
  x <- rnorm(100); y <- 0.4 * x + rnorm(100)
  cat("\n=== Correlation permutation ===\n"); print(perm_correlation(x, y))
  n <- 200; X <- cbind(1, rnorm(n), rnorm(n))
  y2 <- 1 + 0.5 * X[, 2] + rnorm(n, 0, 0.5)
  cat("\n=== Regression coef permutation ===\n")
  print(perm_regression_coef(X, y2, coef_index = 2))
}
