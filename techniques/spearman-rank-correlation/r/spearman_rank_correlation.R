# Spearman rank correlation (Reference §4.2)
# From-scratch base R plus stats::cor / stats::cor.test (method = "spearman").
# Run with:  Rscript spearman_rank_correlation.R
#
# Inputs used below:
#   x, y        : paired numeric vectors of equal length
#   alternative : "two.sided" / "less" / "greater"
#   conf        : confidence level for the Fisher-z CI on rho

spearman_rho_scratch <- function(x, y) {
  stopifnot(length(x) == length(y))
  # base R's rank() handles ties via average by default -- exactly what we want
  rx <- rank(x); ry <- rank(y)
  mx <- mean(rx); my <- mean(ry)
  sxy <- sum((rx - mx) * (ry - my))
  sxx <- sum((rx - mx)^2); syy <- sum((ry - my)^2)
  sxy / sqrt(sxx * syy)
}

spearman_test_scratch <- function(x, y, alternative = "two.sided", conf = 0.95) {
  n <- length(x); r <- spearman_rho_scratch(x, y)
  if (abs(r) == 1) return(list(rho = r, p_value = 0, ci_lower = r, ci_upper = r,
                               method = "Spearman"))
  t <- r * sqrt((n - 2) / (1 - r^2)); df <- n - 2
  p <- switch(alternative,
    "two.sided" = 2 * pt(-abs(t), df),
    "greater"   = pt(t, df, lower.tail = FALSE),
    "less"      = pt(t, df))
  z <- atanh(r); se <- 1 / sqrt(n - 3); zc <- qnorm(0.5 + conf / 2)
  list(rho = r, t = t, df = df, p_value = p,
       ci_lower = tanh(z - zc * se), ci_upper = tanh(z + zc * se),
       method = "Spearman", alternative = alternative)
}

# Library: stats::cor(x, y, method = "spearman"), stats::cor.test(...)
library_demo <- function(x, y) {
  cat("cor(x, y, method = 'spearman'):", cor(x, y, method = "spearman"), "\n")
  print(cor.test(x, y, method = "spearman", exact = FALSE))
}

if (sys.nframe() == 0) {
  set.seed(6); n <- 60
  x <- rnorm(n); y <- exp(x) + rnorm(n, 0, 0.3)
  cat("=== Spearman on n =", n, " (monotone, nonlinear) ===\n")
  print(spearman_test_scratch(x, y))
  cat("\nPearson r for contrast:", cor(x, y), "\n")
  cat("\n--- library ---\n"); library_demo(x, y)
}
