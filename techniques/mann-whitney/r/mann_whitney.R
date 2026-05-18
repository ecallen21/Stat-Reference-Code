# Mann-Whitney U / Wilcoxon Rank-Sum Test (Reference §6.3)
# From-scratch base R plus stats::wilcox.test.
# Run with:  Rscript mann_whitney.R
#
# Inputs:
#   x1, x2      : two independent numeric samples
#   alternative : "two.sided" / "greater" / "less"

mann_whitney_scratch <- function(x1, x2, alternative = "two.sided") {
  n1 <- length(x1); n2 <- length(x2); n <- n1 + n2
  ranks <- rank(c(x1, x2), ties.method = "average")
  R1 <- sum(ranks[seq_len(n1)]); R2 <- sum(ranks[(n1 + 1):n])
  U1 <- R1 - n1 * (n1 + 1) / 2; U2 <- n1 * n2 - U1
  mu <- n1 * n2 / 2
  ties <- table(c(x1, x2)); tie_corr <- sum(ties^3 - ties) / (n * (n - 1))
  var <- n1 * n2 / 12 * (n + 1 - tie_corr)
  p <- switch(alternative,
    "two.sided" = {
      z <- (U1 - mu - 0.5 * sign(U1 - mu)) / sqrt(var)
      2 * pnorm(-abs(z))
    },
    "greater" = pnorm((U1 - mu - 0.5) / sqrt(var), lower.tail = FALSE),
    "less"    = pnorm((U1 - mu + 0.5) / sqrt(var)))
  rank_biserial <- 2 * U1 / (n1 * n2) - 1
  list(U1 = U1, U2 = U2, R1 = R1, R2 = R2, n1 = n1, n2 = n2,
       p_value = p, rank_biserial = rank_biserial,
       method = "Mann-Whitney U (normal approx)")
}

# Library: stats::wilcox.test
library_demo <- function(x1, x2)
  print(wilcox.test(x1, x2, alternative = "two.sided", exact = FALSE))

if (sys.nframe() == 0) {
  set.seed(2)
  a <- rnorm(30, 50, 10); b <- rnorm(28, 55, 12)
  print(mann_whitney_scratch(a, b))
  cat("\n--- library ---\n"); library_demo(a, b)
}
