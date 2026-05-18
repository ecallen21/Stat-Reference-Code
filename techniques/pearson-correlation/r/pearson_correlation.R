# Pearson product-moment correlation (Reference §4.1)
# From-scratch base R plus stats::cor / stats::cor.test.
# Run with:  Rscript pearson_correlation.R
#
# Inputs used below:
#   x, y        : paired numeric vectors of equal length
#   alternative : "two.sided" / "less" / "greater"
#   conf        : confidence level for the Fisher-z CI on rho

pearson_r_scratch <- function(x, y) {
  stopifnot(length(x) == length(y))
  mx <- mean(x); my <- mean(y)
  sxy <- sum((x - mx) * (y - my))
  sxx <- sum((x - mx)^2); syy <- sum((y - my)^2)
  sxy / sqrt(sxx * syy)
}

pearson_test_scratch <- function(x, y, alternative = "two.sided", conf = 0.95) {
  n <- length(x); r <- pearson_r_scratch(x, y)
  if (abs(r) == 1) return(list(r = r, t = Inf, df = n - 2, p_value = 0,
                               ci_lower = r, ci_upper = r, method = "Pearson"))
  t <- r * sqrt((n - 2) / (1 - r^2)); df <- n - 2
  p <- switch(alternative,
    "two.sided" = 2 * pt(-abs(t), df),
    "greater"   = pt(t, df, lower.tail = FALSE),
    "less"      = pt(t, df))
  z <- atanh(r); se <- 1 / sqrt(n - 3); zc <- qnorm(0.5 + conf / 2)
  list(r = r, t = t, df = df, p_value = p,
       ci_lower = tanh(z - zc * se), ci_upper = tanh(z + zc * se),
       method = "Pearson", alternative = alternative)
}

# Library: stats::cor (point estimate), stats::cor.test (test + CI)
library_demo <- function(x, y) {
  cat("cor(x, y, method = 'pearson'):", cor(x, y), "\n")
  print(cor.test(x, y, method = "pearson"))
}

if (sys.nframe() == 0) {
  set.seed(5); n <- 80
  x <- rnorm(n); y <- 0.6 * x + sqrt(1 - 0.6^2) * rnorm(n)
  cat("=== Pearson on n =", n, "===\n"); print(pearson_test_scratch(x, y))
  cat("\n--- library ---\n"); library_demo(x, y)

  cat("\n=== zero r, quadratic relationship ===\n")
  x2 <- seq(-3, 3, length.out = 100); y2 <- x2^2 + rnorm(100, 0, 0.1)
  cat("r =", pearson_r_scratch(x2, y2), "\n")
}
