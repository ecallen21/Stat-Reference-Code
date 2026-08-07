# TMLE + doubly-robust ATE (Reference §15.11)
# R via tmle (van der Laan) or SuperLearner-backed lmtp.
# Run with:  Rscript tmle_doubly_robust.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 1000
  w1 <- rnorm(n); w2 <- rnorm(n)
  A <- as.integer(runif(n) < plogis(-0.5 + w1 - 0.5 * w2))
  Y <- 1 + 2 * A + 0.7 * w1 - 0.4 * w2 + 0.3 * A * w1 + rnorm(n)
  cat("=== Naive ===\n")
  cat(sprintf("  %.3f\n", mean(Y[A == 1]) - mean(Y[A == 0])))
  if (requireNamespace("tmle", quietly = TRUE)) {
    cat("\n=== tmle::tmle (uses SuperLearner by default) ===\n")
    fit <- tmle::tmle(Y = Y, A = A, W = data.frame(w1 = w1, w2 = w2),
                       Q.SL.library = c("SL.glm"), g.SL.library = c("SL.glm"))
    print(summary(fit))
  }
}
