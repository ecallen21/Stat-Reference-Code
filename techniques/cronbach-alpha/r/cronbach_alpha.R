# Cronbach alpha + McDonald omega (Reference §22.3)
# R via psych::alpha and psych::omega.
# Run with:  Rscript cronbach_alpha.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 300; K <- 6
  theta <- rnorm(n)
  lam <- c(0.7, 0.7, 0.7, 0.8, 0.6, 0.65)
  X <- theta %*% t(lam) + matrix(rnorm(n * K, 0, 0.5), n, K)
  if (requireNamespace("psych", quietly = TRUE)) {
    cat("=== psych::alpha ===\n")
    print(psych::alpha(X))
    cat("\n=== psych::omega ===\n")
    print(psych::omega(X, nfactors = 1))
  }
}
