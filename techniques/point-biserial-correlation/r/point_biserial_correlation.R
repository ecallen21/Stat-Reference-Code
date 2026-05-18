# Point-biserial correlation (Reference §4.4)
# From-scratch base R plus stats::cor.test (a Pearson cor.test on 0/1 vs. continuous
# returns exactly the point-biserial correlation).
# Run with:  Rscript point_biserial_correlation.R
#
# Inputs used below:
#   x : binary integer/logical vector coded 0 / 1
#   y : continuous numeric vector of the same length

point_biserial_scratch <- function(x, y) {
  stopifnot(length(x) == length(y), all(x %in% c(0, 1)))
  n <- length(x); n1 <- sum(x == 1); n0 <- sum(x == 0)
  stopifnot(n1 > 0, n0 > 0)
  m1 <- mean(y[x == 1]); m0 <- mean(y[x == 0])
  p <- n1 / n
  sy <- sqrt(sum((y - mean(y))^2) / n)   # population SD, matches scipy
  r_pb <- (m1 - m0) / sy * sqrt(p * (1 - p))
  t <- r_pb * sqrt((n - 2) / (1 - r_pb^2)); df <- n - 2
  list(r_pb = r_pb, mean_group_1 = m1, mean_group_0 = m0,
       n1 = n1, n0 = n0, t = t, df = df,
       p_value = 2 * pt(-abs(t), df))
}

# Library: cor.test(x, y) on a 0/1 x is exactly the point-biserial correlation
library_demo <- function(x, y) {
  cat("cor.test(x, y):\n"); print(cor.test(x, y))
  cat("\nEquivalent two-sample t-test:\n"); print(t.test(y[x == 1], y[x == 0], var.equal = TRUE))
}

if (sys.nframe() == 0) {
  set.seed(8); n <- 100
  x <- sample(0:1, n, replace = TRUE)
  y <- ifelse(x == 0, rnorm(n, 50, 10), rnorm(n, 56, 10))
  cat("=== Point-biserial ===\n"); print(point_biserial_scratch(x, y))
  cat("\n--- library ---\n"); library_demo(x, y)
}
