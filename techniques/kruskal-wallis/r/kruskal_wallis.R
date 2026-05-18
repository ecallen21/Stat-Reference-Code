# Kruskal-Wallis Test (Reference §6.4)
# From-scratch base R plus stats::kruskal.test.
# Run with:  Rscript kruskal_wallis.R
#
# Inputs:  groups -- list of numeric vectors (one per group)

kruskal_wallis_scratch <- function(groups) {
  all_vals <- unlist(groups); sizes <- vapply(groups, length, integer(1))
  N <- length(all_vals); k <- length(groups)
  ranks <- rank(all_vals, ties.method = "average")
  rank_sums <- sapply(seq_along(groups),
                      function(i) sum(ranks[(sum(sizes[seq_len(i - 1)]) + 1):sum(sizes[seq_len(i)])]))
  H <- (12 / (N * (N + 1))) * sum(rank_sums^2 / sizes) - 3 * (N + 1)
  ties <- table(all_vals); tie_sum <- sum(ties^3 - ties)
  H_corr <- if (N^3 - N > 0) H / (1 - tie_sum / (N^3 - N)) else H
  list(H = H, H_corrected = H_corr, df = k - 1,
       p_value = pchisq(H_corr, k - 1, lower.tail = FALSE),
       n_per_group = sizes, rank_sums = rank_sums, method = "Kruskal-Wallis")
}

library_demo <- function(groups) {
  values <- unlist(groups)
  grp <- factor(rep(seq_along(groups), sapply(groups, length)))
  print(kruskal.test(values ~ grp))
}

if (sys.nframe() == 0) {
  set.seed(3)
  a <- rnorm(30, 50, 8); b <- rnorm(28, 55, 9); c <- rnorm(32, 60, 10)
  print(kruskal_wallis_scratch(list(a, b, c)))
  cat("\n--- library ---\n"); library_demo(list(a, b, c))
}
