# Hodges-Lehmann Estimator (Reference §6.29)
# From-scratch base R plus stats::wilcox.test(conf.int = TRUE) which returns HL.
# Run with:  Rscript hodges_lehmann.R

hodges_lehmann_one_sample_scratch <- function(x) {
  n <- length(x)
  walsh <- as.vector(outer(x, x, "+")) / 2
  walsh <- walsh[upper.tri(matrix(0, n, n), diag = TRUE)]
  median(walsh)
}

hodges_lehmann_two_sample_scratch <- function(x, y)
  median(as.vector(outer(x, y, "-")))

hodges_lehmann_ci_two_sample_scratch <- function(x, y, conf = 0.95) {
  n1 <- length(x); n2 <- length(y)
  diffs <- sort(as.vector(outer(x, y, "-")))
  mu <- n1 * n2 / 2; var <- n1 * n2 * (n1 + n2 + 1) / 12
  z <- qnorm(0.5 + conf / 2); half <- z * sqrt(var)
  k <- max(0, floor(mu - half))
  list(estimate = median(diffs),
       ci_lower = diffs[k + 1],          # 1-indexed
       ci_upper = diffs[length(diffs) - k],
       conf = conf)
}

library_demo <- function(x, y) {
  cat("wilcox.test(x, y, conf.int = TRUE)  --  returns HL estimate and CI:\n")
  print(wilcox.test(x, y, conf.int = TRUE))
}

if (sys.nframe() == 0) {
  set.seed(10)
  x <- rnorm(25, 5, 2); y <- rnorm(30, 6.5, 2.5)
  cat("One-sample HL :", hodges_lehmann_one_sample_scratch(x), "\n")
  cat("Two-sample HL :", hodges_lehmann_two_sample_scratch(x, y), "\n")
  print(hodges_lehmann_ci_two_sample_scratch(x, y))
  cat("\n--- library ---\n"); library_demo(x, y)
}
