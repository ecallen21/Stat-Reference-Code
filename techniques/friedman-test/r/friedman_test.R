# Friedman Test (Reference §6.5)
# From-scratch base R plus stats::friedman.test.
# Run with:  Rscript friedman_test.R
#
# Inputs:  data -- n x k matrix (rows = subjects, cols = treatments)

friedman_test_scratch <- function(data) {
  n <- nrow(data); k <- ncol(data)
  ranks <- t(apply(data, 1, rank, ties.method = "average"))
  R <- colSums(ranks)
  F <- (12 / (n * k * (k + 1))) * sum(R^2) - 3 * n * (k + 1)
  ties_per_row <- apply(data, 1, function(r) {
    counts <- table(r); sum(counts^3 - counts)
  })
  denom <- 1 - sum(ties_per_row) / (n * (k^3 - k))
  F_corr <- if (denom != 0) F / denom else F
  list(F = F, F_corrected = F_corr, df = k - 1,
       p_value = pchisq(F_corr, k - 1, lower.tail = FALSE),
       rank_sums = R,
       kendalls_W = F_corr / (n * (k - 1)),
       n_subjects = n, k_treatments = k, method = "Friedman")
}

library_demo <- function(data) print(friedman.test(data))

if (sys.nframe() == 0) {
  set.seed(4); n <- 20; k <- 4
  subj <- rnorm(n, 50, 8)
  effects <- c(0, 2, 5, 1)
  data <- outer(subj, rep(1, k)) + matrix(effects, n, k, byrow = TRUE) + rnorm(n * k, 0, 2)
  print(friedman_test_scratch(data))
  cat("\n--- library ---\n"); library_demo(data)
}
