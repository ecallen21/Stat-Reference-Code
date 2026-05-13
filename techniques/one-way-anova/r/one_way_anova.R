# One-way ANOVA: classic, Welch, Brown-Forsythe (Reference §3.8, §3.9)
# From-scratch base R plus stats::oneway.test and stats::aov.
# Run with:  Rscript one_way_anova.R
#
# Inputs used below:
#   groups : list of numeric vectors, one per group (length k)

stats_of <- function(g) c(n = length(g), m = mean(g), s2 = var(g))

classic_anova_scratch <- function(groups) {
  s <- vapply(groups, stats_of, numeric(3))
  ns <- s["n",]; means <- s["m",]; vars <- s["s2",]
  N <- sum(ns); k <- length(ns)
  grand <- sum(ns * means) / N
  ss_b <- sum(ns * (means - grand)^2)
  ss_w <- sum((ns - 1) * vars)
  df_b <- k - 1; df_w <- N - k
  ms_b <- ss_b / df_b; ms_w <- ss_w / df_w
  F <- ms_b / ms_w; p <- pf(F, df_b, df_w, lower.tail = FALSE)
  eta2 <- ss_b / (ss_b + ss_w)
  omega2 <- (ss_b - df_b * ms_w) / (ss_b + ss_w + ms_w)
  list(k = k, N = N, ss_between = ss_b, ss_within = ss_w,
       df1 = df_b, df2 = df_w, ms_between = ms_b, ms_within = ms_w,
       F = F, p_value = p, eta_squared = eta2, omega_squared = omega2)
}

welch_anova_scratch <- function(groups) {
  s <- vapply(groups, stats_of, numeric(3))
  ns <- s["n",]; means <- s["m",]; vars <- s["s2",]; k <- length(ns)
  w <- ns / vars; W <- sum(w)
  grand <- sum(w * means) / W
  num <- sum(w * (means - grand)^2) / (k - 1)
  denom <- 1 + (2 * (k - 2) / (k^2 - 1)) *
                sum((1 - w / W)^2 / (ns - 1))
  F <- num / denom
  df1 <- k - 1
  df2 <- (k^2 - 1) / (3 * sum((1 - w / W)^2 / (ns - 1)))
  list(F = F, df1 = df1, df2 = df2,
       p_value = pf(F, df1, df2, lower.tail = FALSE), method = "Welch")
}

brown_forsythe_scratch <- function(groups) {
  s <- vapply(groups, stats_of, numeric(3))
  ns <- s["n",]; means <- s["m",]; vars <- s["s2",]
  N <- sum(ns); k <- length(ns); grand <- sum(ns * means) / N
  num <- sum(ns * (means - grand)^2)
  denom <- sum((1 - ns / N) * vars)
  F <- num / denom
  df1 <- k - 1
  df2 <- denom^2 / sum(((1 - ns / N) * vars)^2 / (ns - 1))
  list(F = F, df1 = df1, df2 = df2,
       p_value = pf(F, df1, df2, lower.tail = FALSE), method = "Brown-Forsythe")
}

# Library: stats::oneway.test (Welch by default; classic with var.equal = TRUE),
#          stats::aov + summary for the classic ANOVA table
library_demo <- function(groups) {
  values <- unlist(groups)
  grp <- factor(rep(seq_along(groups), sapply(groups, length)))
  cat("aov (classic):\n"); print(summary(aov(values ~ grp)))
  cat("\noneway.test Welch:\n"); print(oneway.test(values ~ grp, var.equal = FALSE))
  cat("\noneway.test classic:\n"); print(oneway.test(values ~ grp, var.equal = TRUE))
}

if (sys.nframe() == 0) {
  set.seed(2)
  a <- rnorm(30, 50, 10); b <- rnorm(28, 55, 10); c <- rnorm(32, 60, 18)
  cat("=== classic ANOVA ===\n");    print(classic_anova_scratch(list(a, b, c)))
  cat("\n=== Welch ANOVA ===\n");    print(welch_anova_scratch(list(a, b, c)))
  cat("\n=== Brown-Forsythe ===\n"); print(brown_forsythe_scratch(list(a, b, c)))
  cat("\n--- library ---\n");        library_demo(list(a, b, c))
}
