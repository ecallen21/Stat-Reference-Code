# Wilcoxon Signed-Rank Test (Reference §6.2)
# From-scratch base R plus stats::wilcox.test.
# Run with:  Rscript wilcoxon_signed_rank.R
#
# Inputs:
#   x, m0       : sample and null median (one-sample form)
#   x1, x2      : paired samples (apply to differences)
#   alternative : "two.sided" / "greater" / "less"

wilcoxon_signed_rank_scratch <- function(x, m0 = 0, alternative = "two.sided") {
  d <- x[x != m0] - m0; n_eff <- length(d)
  if (n_eff == 0) return(list(W_plus = 0, n_effective = 0, p_value = 1))
  abs_d <- abs(d); ranks <- rank(abs_d, ties.method = "average")
  W_pos <- sum(ranks[d > 0]); W_neg <- sum(ranks[d < 0])
  mu <- n_eff * (n_eff + 1) / 4
  ties <- table(abs_d); tie_corr <- sum(ties^3 - ties) / 48
  var <- n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24 - tie_corr
  p <- switch(alternative,
    "two.sided" = {
      z <- (W_pos - mu - 0.5 * sign(W_pos - mu)) / sqrt(var)
      2 * pnorm(-abs(z))
    },
    "greater" = {
      z <- (W_pos - mu - 0.5) / sqrt(var); pnorm(z, lower.tail = FALSE)
    },
    "less" = {
      z <- (W_pos - mu + 0.5) / sqrt(var); pnorm(z)
    })
  list(W_plus = W_pos, W_neg = W_neg, n_effective = n_eff,
       p_value = p, method = "Wilcoxon signed-rank (normal approx)")
}

paired_wilcoxon_scratch <- function(x1, x2, alternative = "two.sided") {
  stopifnot(length(x1) == length(x2))
  wilcoxon_signed_rank_scratch(x1 - x2, m0 = 0, alternative = alternative)
}

library_demo <- function(x, m0) {
  print(wilcox.test(x - m0, alternative = "two.sided", exact = FALSE))
}

if (sys.nframe() == 0) {
  set.seed(1)
  cat("=== One-sample, H0: median = 50 ===\n")
  x <- rnorm(30, 52, 8); print(wilcoxon_signed_rank_scratch(x, 50))
  cat("\n=== Paired ===\n")
  pre <- rnorm(25, 100, 10); post <- pre + rnorm(25, 3, 4)
  print(paired_wilcoxon_scratch(post, pre))
  cat("\n--- library ---\n"); library_demo(x, 50)
}
