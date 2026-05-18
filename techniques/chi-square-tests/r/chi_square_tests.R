# Chi-square tests (Reference §3.5)
# From-scratch base R plus stats::chisq.test (idiomatic for both GOF and independence).
# Run with:  Rscript chi_square_tests.R
#
# Inputs used below:
#   observed       : integer vector of observed counts per category (GOF)
#   expected_probs : numeric vector of null probabilities (sum to 1); default uniform
#   ddof           : number of parameters estimated from the data (subtract from df)
#   table          : r x c matrix of counts (rows of contingency table)
#   correction     : Yates' continuity correction (2x2 only)

goodness_of_fit_scratch <- function(observed, expected_probs = NULL, ddof = 0) {
  n <- sum(observed); k <- length(observed)
  if (is.null(expected_probs)) expected_probs <- rep(1/k, k)
  stopifnot(abs(sum(expected_probs) - 1) < 1e-9)
  expected <- n * expected_probs
  chi2 <- sum((observed - expected)^2 / expected)
  df <- k - 1 - ddof
  list(observed = observed, expected = expected,
       chi_square = chi2, df = df, p_value = pchisq(chi2, df, lower.tail = FALSE))
}

independence_scratch <- function(table, correction = FALSE) {
  r <- nrow(table); c <- ncol(table)
  row_tot <- rowSums(table); col_tot <- colSums(table); n <- sum(table)
  expected <- outer(row_tot, col_tot) / n
  if (correction && r == 2 && c == 2) {
    chi2 <- sum(pmax(0, abs(table - expected) - 0.5)^2 / expected)
  } else {
    chi2 <- sum((table - expected)^2 / expected)
  }
  df <- (r - 1) * (c - 1)
  p <- pchisq(chi2, df, lower.tail = FALSE)
  cramers_v <- sqrt(chi2 / (n * min(r - 1, c - 1)))
  residuals <- (table - expected) / sqrt(expected)
  list(row_totals = row_tot, col_totals = col_tot, n = n,
       expected = expected, chi_square = chi2, df = df, p_value = p,
       cramers_v = cramers_v, residuals = residuals,
       warning_small_expected = any(expected < 5))
}

# Library: stats::chisq.test handles both GOF and independence
library_demo <- function(rolls, table) {
  cat("GOF (equal probs):\n"); print(chisq.test(rolls))
  cat("\nIndependence (no correction):\n"); print(chisq.test(table, correct = FALSE))
  cat("\nStandardized residuals:\n"); print(chisq.test(table, correct = FALSE)$stdres)
}

if (sys.nframe() == 0) {
  cat("=== GOF: die fairness ===\n"); print(goodness_of_fit_scratch(c(18,22,19,16,23,22)))
  tbl <- matrix(c(30, 25, 10, 10, 15, 20), nrow = 3, byrow = FALSE,
                dimnames = list(c("North","South","East"), c("Yes","No")))
  cat("\n=== Independence: region x outcome ===\n"); print(independence_scratch(tbl))
  cat("\n=== 2x2 with Yates correction ===\n")
  print(independence_scratch(matrix(c(8, 15, 12, 5), nrow = 2), correction = TRUE))
  cat("\n--- library (stats::chisq.test) ---\n"); library_demo(c(18,22,19,16,23,22), tbl)
}
