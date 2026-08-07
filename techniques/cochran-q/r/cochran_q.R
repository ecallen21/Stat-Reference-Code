# Cochran's Q test (Reference §8.10)
# Base R has no direct function; use nonpar::cochran.q or manual.
# Run with:  Rscript cochran_q.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 30; k <- 4
  p_true <- c(0.30, 0.35, 0.55, 0.60)
  u <- rnorm(n, 0, 0.6)
  logit <- log(p_true / (1 - p_true))
  p_ij <- outer(u, logit, "+"); p_ij <- 1 / (1 + exp(-p_ij))
  X <- (matrix(runif(n * k), n, k) < p_ij) * 1
  if (requireNamespace("nonpar", quietly = TRUE)) {
    cat("=== nonpar::cochran.q ===\n")
    print(nonpar::cochran.q(X))
  } else {
    R <- rowSums(X); C <- colSums(X)
    Q <- (k - 1) * (k * sum(C^2) - sum(C)^2) / (k * sum(R) - sum(R^2))
    cat(sprintf("Q = %.4f, df = %d, p = %.4f\n", Q, k - 1, pchisq(Q, k - 1, lower.tail = FALSE)))
  }
}
