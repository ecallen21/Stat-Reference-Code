# z-tests for means and proportions (Reference §3.7, §3.22)
# From-scratch base R plus stats::prop.test (the canonical proportion z-test).
# Run with:  Rscript z_tests.R
#
# Inputs used below:
#   x, x1, x2   : numeric vectors (the samples; means)  OR  integer counts (proportions)
#   n, n1, n2   : sample sizes (proportions)
#   mu0, p0     : null hypothesis value for the mean / proportion
#   sigma, sigma1, sigma2 : KNOWN population SDs (for the mean tests)
#   alternative : "two.sided" / "less" / "greater"
#   conf        : confidence level
#   continuity  : TRUE -> Yates continuity correction

z_pvalue <- function(z, alternative) switch(alternative,
  "two.sided" = 2 * pnorm(-abs(z)),
  "greater"   = pnorm(z, lower.tail = FALSE),
  "less"      = pnorm(z),
  stop("alternative must be 'two.sided', 'less', or 'greater'"))

z_ci <- function(est, se, conf) {
  zc <- qnorm(0.5 + conf / 2); c(est - zc * se, est + zc * se)
}

one_sample_mean_z <- function(x, mu0, sigma, alternative = "two.sided", conf = 0.95) {
  n <- length(x); m <- mean(x); se <- sigma / sqrt(n); z <- (m - mu0) / se
  ci <- z_ci(m, se, conf)
  list(mean = m, se = se, z = z, p_value = z_pvalue(z, alternative),
       ci_lower = ci[1], ci_upper = ci[2])
}

two_sample_mean_z <- function(x1, x2, sigma1, sigma2,
                              alternative = "two.sided", conf = 0.95) {
  n1 <- length(x1); n2 <- length(x2); diff <- mean(x1) - mean(x2)
  se <- sqrt(sigma1^2 / n1 + sigma2^2 / n2); z <- diff / se
  ci <- z_ci(diff, se, conf)
  list(mean_diff = diff, se = se, z = z, p_value = z_pvalue(z, alternative),
       ci_lower = ci[1], ci_upper = ci[2])
}

one_proportion_z <- function(x, n, p0, alternative = "two.sided",
                             conf = 0.95, continuity = FALSE) {
  p_hat <- x / n; se0 <- sqrt(p0 * (1 - p0) / n); d <- abs(p_hat - p0)
  if (continuity) d <- max(0, d - 1 / (2 * n))
  z <- sign(p_hat - p0) * d / se0
  ci <- z_ci(p_hat, sqrt(p_hat * (1 - p_hat) / n), conf)
  list(p_hat = p_hat, se_null = se0, z = z,
       p_value = z_pvalue(z, alternative),
       ci_lower_wald = max(0, ci[1]), ci_upper_wald = min(1, ci[2]))
}

two_proportion_z <- function(x1, n1, x2, n2, alternative = "two.sided",
                             conf = 0.95, continuity = FALSE) {
  p1 <- x1 / n1; p2 <- x2 / n2; diff <- p1 - p2
  p_pool <- (x1 + x2) / (n1 + n2)
  se_p <- sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
  if (continuity) {
    cc <- 0.5 * (1/n1 + 1/n2); adj <- sign(diff) * max(0, abs(diff) - cc)
  } else adj <- diff
  z <- adj / se_p
  se_u <- sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
  ci <- z_ci(diff, se_u, conf)
  list(p1 = p1, p2 = p2, diff = diff, p_pool = p_pool, se_pooled = se_p,
       z = z, p_value = z_pvalue(z, alternative),
       ci_lower = ci[1], ci_upper = ci[2])
}

# Library: stats::prop.test is the chi-square (= z^2) version with continuity correction
library_demo <- function(x1, n1, x2, n2) {
  cat("prop.test (two-proportion, correct = FALSE):\n")
  print(prop.test(c(x1, x2), c(n1, n2), correct = FALSE))
}

if (sys.nframe() == 0) {
  set.seed(1)
  x <- rnorm(50, 101, 15)
  cat("=== one-sample mean z (mu0 = 100, sigma = 15) ===\n")
  print(one_sample_mean_z(x, mu0 = 100, sigma = 15))
  a <- rnorm(60, 105, 12); b <- rnorm(55, 100, 18)
  cat("\n=== two-sample mean z (sigma1 = 12, sigma2 = 18) ===\n")
  print(two_sample_mean_z(a, b, 12, 18))
  cat("\n=== one-proportion z (42 / 100, p0 = 0.5) ===\n")
  print(one_proportion_z(42, 100, 0.5))
  cat("\n=== two-proportion z (42/100 vs 30/100) ===\n")
  print(two_proportion_z(42, 100, 30, 100))
  cat("\n--- library (stats::prop.test) ---\n"); library_demo(42, 100, 30, 100)
}
