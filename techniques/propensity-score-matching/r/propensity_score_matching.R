# Propensity Score Matching (Reference §15.6)
# R via MatchIt (Ho-Imai-King-Stuart).
# Run with:  Rscript propensity_score_matching.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 800
  x1 <- rnorm(n); x2 <- rnorm(n)
  T <- as.integer(runif(n) < plogis(-0.5 + 1 * x1 - 0.5 * x2))
  Y <- 1 + 2 * T + 0.5 * x1 + 0.3 * x2 + rnorm(n)
  df <- data.frame(T = T, x1 = x1, x2 = x2, Y = Y)
  cat("=== Naive diff in means (biased) ===\n")
  cat(sprintf("  %.3f\n", mean(Y[T == 1]) - mean(Y[T == 0])))
  if (requireNamespace("MatchIt", quietly = TRUE)) {
    cat("\n=== MatchIt 1:1 NN with replacement ===\n")
    m <- MatchIt::matchit(T ~ x1 + x2, data = df, method = "nearest", replace = TRUE)
    print(summary(m))
    md <- MatchIt::match.data(m)
    cat(sprintf("\n  ATT (matched diff) = %.3f\n",
                mean(md$Y[md$T == 1]) - mean(md$Y[md$T == 0])))
  }
}
