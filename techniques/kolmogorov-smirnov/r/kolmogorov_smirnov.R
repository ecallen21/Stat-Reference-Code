# Kolmogorov-Smirnov Test (Reference §6.7)
# From-scratch base R plus stats::ks.test.
# Run with:  Rscript kolmogorov_smirnov.R
#
# Inputs:
#   x, y : numeric samples
#   cdf  : a function giving the target CDF (one-sample form)

one_sample_ks_scratch <- function(x, cdf) {
  x <- sort(x); n <- length(x)
  d_plus <- max(seq_len(n) / n - cdf(x))
  d_minus <- max(cdf(x) - (seq_len(n) - 1) / n)
  D <- max(d_plus, d_minus)
  # asymptotic p via the Kolmogorov limiting distribution
  p <- 1 - sapply(1, function(.) {
    z <- sqrt(n) * D
    sum_terms <- vapply(1:50, function(k) ((-1)^(k - 1)) * exp(-2 * (k * z)^2), numeric(1))
    1 - 2 * sum(sum_terms)
  })
  list(D = D, D_plus = d_plus, D_minus = d_minus, n = n,
       p_value_asymptotic = p[1], method = "1-sample KS")
}

two_sample_ks_scratch <- function(x, y) {
  m <- length(x); n <- length(y)
  pts <- sort(unique(c(x, y)))
  Fx <- ecdf(x)(pts); Fy <- ecdf(y)(pts)
  D <- max(abs(Fx - Fy))
  en <- sqrt(m * n / (m + n))
  z <- en * D
  sum_terms <- vapply(1:50, function(k) ((-1)^(k - 1)) * exp(-2 * (k * z)^2), numeric(1))
  p <- 2 * sum(sum_terms)
  list(D = D, m = m, n = n, p_value_asymptotic = p, method = "2-sample KS")
}

library_demo <- function(x, y) {
  print(ks.test(x, "pnorm", 0, 1))
  print(ks.test(x, y))
}

if (sys.nframe() == 0) {
  set.seed(5)
  x <- rnorm(200, 0, 1); y <- rnorm(180, 0.4, 1)
  cat("=== 1-sample, test vs N(0, 1) ===\n"); print(one_sample_ks_scratch(x, pnorm))
  cat("\n=== 2-sample ===\n"); print(two_sample_ks_scratch(x, y))
  cat("\n--- library ---\n"); library_demo(x, y)
}
