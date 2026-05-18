# Binomial test (Reference §3.22)
# From-scratch base R plus stats::binom.test. Run with:  Rscript binomial_test.R
#
# Inputs used below:
#   x           : observed number of successes
#   n           : number of trials
#   p0          : null hypothesis success probability
#   alternative : "two.sided" / "greater" / "less"
#   continuity  : Yates correction for the normal approximation

pmf <- function(n, k, p) dbinom(k, n, p)

exact_binomial_test <- function(x, n, p0 = 0.5, alternative = "two.sided") {
  stopifnot(0 <= x, x <= n)
  p_hat <- x / n
  p_val <- switch(alternative,
    "greater"   = sum(pmf(n, x:n, p0)),
    "less"      = sum(pmf(n, 0:x, p0)),
    "two.sided" = {
      p_x <- pmf(n, x, p0)
      sum(pmf(n, 0:n, p0)[pmf(n, 0:n, p0) <= p_x + 1e-15])
    },
    stop("alternative must be 'two.sided', 'greater', or 'less'"))
  list(x = x, n = n, p_hat = p_hat, p0 = p0,
       p_value = min(1, p_val), method = "exact")
}

mid_p_binomial_test <- function(x, n, p0 = 0.5, alternative = "two.sided") {
  p_x <- pmf(n, x, p0)
  p_val <- switch(alternative,
    "greater"   = sum(pmf(n, (x + 1):n, p0)) + 0.5 * p_x,
    "less"      = sum(pmf(n, 0:(x - 1), p0)) + 0.5 * p_x,
    "two.sided" = exact_binomial_test(x, n, p0, "two.sided")$p_value - 0.5 * p_x,
    stop("alternative must be 'two.sided', 'greater', or 'less'"))
  list(x = x, n = n, p_hat = x / n, p0 = p0,
       p_value = max(0, p_val), method = "mid-p")
}

normal_approx_binomial_test <- function(x, n, p0 = 0.5,
                                        alternative = "two.sided", continuity = FALSE) {
  p_hat <- x / n; se0 <- sqrt(p0 * (1 - p0) / n)
  diff <- abs(p_hat - p0)
  if (continuity) diff <- max(0, diff - 1 / (2 * n))
  z <- sign(p_hat - p0) * diff / se0
  p_val <- switch(alternative,
    "two.sided" = 2 * pnorm(-abs(z)),
    "greater"   = pnorm(z, lower.tail = FALSE),
    "less"      = pnorm(z))
  list(x = x, n = n, p_hat = p_hat, p0 = p0, z = z,
       p_value = p_val,
       method = paste0("normal-approx", if (continuity) " (Yates)" else ""))
}

# Library: stats::binom.test (exact, two-sided by default), stats::prop.test (normal approx)
library_demo <- function(x, n, p0) {
  cat("stats::binom.test (exact):\n"); print(binom.test(x, n, p0))
  cat("\nstats::prop.test (normal approx):\n"); print(prop.test(x, n, p0))
}

if (sys.nframe() == 0) {
  cat("=== x = 60, n = 100, p0 = 0.5 ===\n")
  for (fn in list(exact_binomial_test, mid_p_binomial_test, normal_approx_binomial_test)) {
    r <- fn(60, 100, 0.5); cat(sprintf("  %-15s: p = %.6f\n", r$method, r$p_value))
  }
  cat("\n=== x = 9, n = 10, p0 = 0.5 (tiny n) ===\n")
  for (fn in list(exact_binomial_test, mid_p_binomial_test,
                  function(x, n, p) normal_approx_binomial_test(x, n, p),
                  function(x, n, p) normal_approx_binomial_test(x, n, p, continuity = TRUE))) {
    r <- fn(9, 10, 0.5); cat(sprintf("  %-22s: p = %.6f\n", r$method, r$p_value))
  }
  cat("\n--- library ---\n"); library_demo(60, 100, 0.5)
}
