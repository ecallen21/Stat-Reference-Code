# Mood's Median Test (Reference §6.8)
# From-scratch base R plus stats::chisq.test on the constructed table,
# or RVAideMemoire::mood.medtest.
# Run with:  Rscript moods_median.R
#
# Inputs:  groups -- list of numeric vectors

moods_median_test_scratch <- function(groups) {
  all_vals <- unlist(groups); M <- median(all_vals)
  tbl <- sapply(groups, function(g) c(above = sum(g > M), below_eq = sum(g <= M)))
  res <- chisq.test(tbl, correct = FALSE)
  list(median = M, table_above_below = tbl,
       chi_square = as.numeric(res$statistic),
       df = as.integer(res$parameter),
       p_value = as.numeric(res$p.value),
       method = "Mood's median")
}

library_demo <- function(groups) {
  if (requireNamespace("RVAideMemoire", quietly = TRUE))
    print(RVAideMemoire::mood.medtest(unlist(groups),
                                       factor(rep(seq_along(groups), sapply(groups, length)))))
  else cat("(install 'RVAideMemoire' for mood.medtest)\n")
}

if (sys.nframe() == 0) {
  set.seed(6)
  a <- rnorm(30, 50, 8); b <- rnorm(28, 55, 9); c <- rnorm(32, 60, 12)
  print(moods_median_test_scratch(list(a, b, c)))
  cat("\n--- library ---\n"); library_demo(list(a, b, c))
}
