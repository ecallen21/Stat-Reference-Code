# Independent Component Analysis via FastICA (Reference §9.9)
# R via fastICA::fastICA.
# Run with:  Rscript independent_components.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 2000; t <- seq(0, 8, length.out = n)
  s1 <- sin(2 * t); s2 <- sign(sin(3 * t)); s3 <- rexp(n) - rexp(n)
  S <- cbind(s1, s2, s3)
  A <- matrix(c(1, 0.5, 0.3, 0.4, 1, 0.5, 0.2, 0.3, 1), 3, 3, byrow = TRUE)
  X <- S %*% t(A)
  if (requireNamespace("fastICA", quietly = TRUE)) {
    cat("=== fastICA::fastICA ===\n")
    fit <- fastICA::fastICA(X, n.comp = 3, alg.typ = "parallel", fun = "logcosh")
    for (k in 1:3) {
      cors <- sapply(1:3, function(j) cor(fit$S[, k], S[, j]))
      cat(sprintf("  component %d best-matches source %d (|corr| = %.3f)\n",
                  k, which.max(abs(cors)), max(abs(cors))))
    }
  }
}
