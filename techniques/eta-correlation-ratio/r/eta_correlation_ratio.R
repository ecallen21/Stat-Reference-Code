# Eta correlation ratio (Reference §4.13)
# Run with:  Rscript eta_correlation_ratio.R
#
# Inputs used below:
#   y_by_group : list of per-group numeric vectors (one entry per category of X)
#   x, y       : parallel categorical / continuous vectors

eta_squared_scratch <- function(y_by_group) {
  all_y <- unlist(y_by_group); N <- length(all_y); k <- length(y_by_group)
  grand <- mean(all_y)
  ss_b <- sum(vapply(y_by_group, function(g) length(g) * (mean(g) - grand)^2, numeric(1)))
  ss_t <- sum((all_y - grand)^2)
  eta2 <- ss_b / ss_t
  df1 <- k - 1; df2 <- N - k
  F <- (eta2 / df1) / ((1 - eta2) / df2)
  list(eta = sqrt(eta2), eta_squared = eta2,
       F = F, df1 = df1, df2 = df2,
       p_value = pf(F, df1, df2, lower.tail = FALSE),
       k_groups = k, N = N)
}

eta_from_columns <- function(x, y) {
  stopifnot(length(x) == length(y))
  groups <- split(y, x)
  eta_squared_scratch(groups)
}

# Library: lsr::etaSquared, effectsize::eta_squared, base R: summary(aov(...))
library_demo <- function(x, y) {
  fit <- aov(y ~ factor(x))
  cat("Base R summary(aov(y ~ x)):\n"); print(summary(fit))
  if (requireNamespace("effectsize", quietly = TRUE)) {
    cat("\neffectsize::eta_squared:\n"); print(effectsize::eta_squared(fit))
  }
}

if (sys.nframe() == 0) {
  set.seed(14)
  g1 <- rnorm(30, 50, 10); g2 <- rnorm(28, 55, 10); g3 <- rnorm(32, 65, 10)
  cat("=== Eta correlation ratio: Y split by 3 groups ===\n")
  print(eta_squared_scratch(list(g1, g2, g3)))

  x <- c(rep("A", 30), rep("B", 28), rep("C", 32))
  y <- c(g1, g2, g3)
  cat("\n=== Same data as columns (x, y) ===\n")
  print(eta_from_columns(x, y))
  cat("\n--- library ---\n"); library_demo(x, y)
}
