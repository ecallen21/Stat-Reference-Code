# Fisher's exact test (Reference §3.6)
# From-scratch base R plus stats::fisher.test. Run with:  Rscript fisher_exact.R
#
# Inputs used below:
#   a, b, c, d  : the four 2x2 cell counts (row-major: top-left, top-right, bottom-left, bottom-right)
#   alternative : "two.sided" / "greater" / "less"
#   conf        : confidence level for the Wald log-OR CI on the OR

hypergeom_pmf <- function(a, n1, n2, k) {
  choose(n1, a) * choose(n2, k - a) / choose(n1 + n2, k)
}

fisher_exact_scratch <- function(a, b, c, d, alternative = "two.sided", conf = 0.95) {
  n1 <- a + b; n2 <- c + d; k <- a + c
  a_min <- max(0, k - n2); a_max <- min(n1, k)
  ai <- a_min:a_max
  pmf <- vapply(ai, hypergeom_pmf, numeric(1), n1 = n1, n2 = n2, k = k)
  p_obs <- pmf[ai == a]

  p_value <- switch(alternative,
    "two.sided" = sum(pmf[pmf <= p_obs + 1e-12]),
    "greater"   = sum(pmf[ai >= a]),
    "less"      = sum(pmf[ai <= a]),
    stop("alternative must be 'two.sided', 'greater', or 'less'"))

  # Haldane-Anscombe correction for zero cells
  if (any(c(a, b, c, d) == 0)) {
    aa <- a + 0.5; bb <- b + 0.5; cc <- c + 0.5; dd <- d + 0.5
  } else { aa <- a; bb <- b; cc <- c; dd <- d }
  or_hat <- (aa * dd) / (bb * cc)
  se <- sqrt(1/aa + 1/bb + 1/cc + 1/dd)
  z <- qnorm(0.5 + conf / 2)
  list(table = matrix(c(a, c, b, d), 2),
       p_observed = p_obs, p_value = min(1, p_value),
       odds_ratio = or_hat, log_or_se = se,
       ci_lower = exp(log(or_hat) - z * se),
       ci_upper = exp(log(or_hat) + z * se),
       alternative = alternative)
}

# Library: stats::fisher.test gives an exact "conditional" CI on the OR
library_demo <- function(a, b, c, d) {
  cat("stats::fisher.test:\n")
  print(fisher.test(matrix(c(a, c, b, d), 2)))
}

if (sys.nframe() == 0) {
  cat("=== 2x2 = [[8, 2], [3, 7]] ===\n")
  print(fisher_exact_scratch(8, 2, 3, 7))
  cat("\n=== one-sided 'greater' (OR > 1) ===\n")
  print(fisher_exact_scratch(8, 2, 3, 7, alternative = "greater"))
  cat("\n=== with a zero cell [[0, 9], [6, 3]] ===\n")
  print(fisher_exact_scratch(0, 9, 6, 3))
  cat("\n--- library (stats::fisher.test) ---\n"); library_demo(8, 2, 3, 7)
}
