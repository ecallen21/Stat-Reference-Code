# Sign Test (Reference §6.1)
# Base R has stats::binom.test (the engine for the sign test).
# Run with:  Rscript sign_test.R
#
# Inputs:
#   x, m0       : sample and null median
#   x1, x2      : paired samples (test the median of the differences)
#   alternative : "two.sided" / "greater" / "less"

sign_test_scratch <- function(x, m0 = 0, alternative = "two.sided") {
  n_pos <- sum(x > m0); n_neg <- sum(x < m0); n_tie <- sum(x == m0)
  n_eff <- n_pos + n_neg
  if (n_eff == 0) return(list(S_pos = 0, n_effective = 0, p_value = 1))
  p_value <- binom.test(n_pos, n_eff, p = 0.5, alternative = alternative)$p.value
  list(S_pos = n_pos, n_neg = n_neg, n_tie = n_tie,
       n_effective = n_eff, p_value = p_value, method = "sign test")
}

paired_sign_test_scratch <- function(x1, x2, alternative = "two.sided") {
  stopifnot(length(x1) == length(x2))
  sign_test_scratch(x1 - x2, m0 = 0, alternative = alternative)
}

# Library: BSDA::SIGN.test (one- and paired-sample sign test)
library_demo <- function(x, m0) {
  if (requireNamespace("BSDA", quietly = TRUE))
    print(BSDA::SIGN.test(x, md = m0, alternative = "greater"))
  else cat("(install 'BSDA' for SIGN.test; the from-scratch matches binom.test)\n")
}

if (sys.nframe() == 0) {
  set.seed(0)
  cat("=== One-sample, median > 50 ===\n")
  x <- rnorm(25, 52, 10); print(sign_test_scratch(x, m0 = 50, alternative = "greater"))
  cat("\n=== Paired ===\n")
  pre <- rnorm(20, 100, 10); post <- pre + rnorm(20, 2, 4)
  print(paired_sign_test_scratch(post, pre, alternative = "greater"))
  cat("\n--- library ---\n"); library_demo(x, 50)
}
