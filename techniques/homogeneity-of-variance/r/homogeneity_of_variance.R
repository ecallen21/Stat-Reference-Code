# Tests for homogeneity of variance across k groups (Reference §3.20, §3.55)
# From-scratch base R plus idiomatic calls (car::leveneTest, stats::bartlett.test,
# stats::var.test).  Run with:  Rscript homogeneity_of_variance.R
#
# Inputs used below:
#   groups : list of numeric vectors (one per group)
#   center : "mean" or "median" (Levene vs. Brown-Forsythe)
#   x1, x2 : two-sample inputs for the F-test of variances

levene_bf <- function(groups, center = "median") {
  k <- length(groups); ns <- vapply(groups, length, integer(1))
  ctr <- vapply(groups, if (center == "median") median else mean, numeric(1))
  z <- mapply(function(g, c0) abs(g - c0), groups, ctr, SIMPLIFY = FALSE)
  N <- sum(ns); grand_z <- mean(unlist(z))
  z_means <- vapply(z, mean, numeric(1))
  ss_b <- sum(ns * (z_means - grand_z)^2)
  ss_w <- sum(unlist(mapply(function(zi, m) (zi - m)^2, z, z_means, SIMPLIFY = FALSE)))
  df1 <- k - 1; df2 <- N - k
  F <- (ss_b / df1) / (ss_w / df2)
  list(statistic = F, df1 = df1, df2 = df2,
       p_value = pf(F, df1, df2, lower.tail = FALSE), center = center)
}

levene_scratch <- function(groups) levene_bf(groups, center = "mean")
brown_forsythe_scratch <- function(groups) levene_bf(groups, center = "median")

bartlett_scratch <- function(groups) {
  k <- length(groups); ns <- vapply(groups, length, integer(1))
  vars <- vapply(groups, var, numeric(1))
  N <- sum(ns); sp2 <- sum((ns - 1) * vars) / (N - k)
  num <- (N - k) * log(sp2) - sum((ns - 1) * log(vars))
  denom <- 1 + (1 / (3 * (k - 1))) * (sum(1 / (ns - 1)) - 1 / (N - k))
  chi2 <- num / denom
  list(statistic = chi2, df = k - 1,
       p_value = pchisq(chi2, k - 1, lower.tail = FALSE))
}

f_test_two_variances_scratch <- function(x1, x2, alternative = "two.sided") {
  v1 <- var(x1); v2 <- var(x2); F <- v1 / v2
  df1 <- length(x1) - 1; df2 <- length(x2) - 1
  p <- switch(alternative,
    "two.sided" = 2 * min(pf(F, df1, df2), pf(F, df1, df2, lower.tail = FALSE)),
    "greater"   = pf(F, df1, df2, lower.tail = FALSE),
    "less"      = pf(F, df1, df2))
  list(F = F, df1 = df1, df2 = df2, p_value = min(1, p), var1 = v1, var2 = v2)
}

# Library: car::leveneTest (with center =), stats::bartlett.test, stats::var.test
library_demo <- function(groups, x1, x2) {
  values <- unlist(groups)
  grp <- factor(rep(seq_along(groups), sapply(groups, length)))
  if (requireNamespace("car", quietly = TRUE)) {
    cat("car::leveneTest (center = median, default):\n")
    print(car::leveneTest(values ~ grp))
  } else cat("(install 'car' for leveneTest)\n")
  cat("\nstats::bartlett.test:\n");  print(bartlett.test(values, grp))
  cat("\nstats::var.test (a vs c):\n"); print(var.test(x1, x2))
}

if (sys.nframe() == 0) {
  set.seed(4)
  a <- rnorm(30, 50, 10); b <- rnorm(28, 50, 11); c <- rnorm(32, 50, 22)
  cat("=== Levene (center = mean) ===\n");        print(levene_scratch(list(a,b,c)))
  cat("\n=== Brown-Forsythe (center = median) ===\n"); print(brown_forsythe_scratch(list(a,b,c)))
  cat("\n=== Bartlett ===\n");                    print(bartlett_scratch(list(a,b,c)))
  cat("\n=== F-test of variances (a vs c) ===\n"); print(f_test_two_variances_scratch(a, c))
  cat("\n--- library ---\n");                     library_demo(list(a,b,c), a, c)
}
