# Procrustes analysis (Reference §9.16)
# Base R via MASS::procrustes / vegan::procrustes.
# Run with:  Rscript procrustes_analysis.R

if (sys.nframe() == 0) {
  set.seed(5); n <- 30
  X <- matrix(rnorm(n * 2), n, 2)
  theta <- pi / 6
  R <- rbind(c(cos(theta), -sin(theta)), c(sin(theta), cos(theta)))
  Y <- 2 * X %*% R + matrix(c(1.5, -0.8), n, 2, byrow = TRUE) + matrix(rnorm(n * 2, 0, 0.05), n, 2)
  if (requireNamespace("vegan", quietly = TRUE)) {
    cat("=== vegan::procrustes ===\n"); print(vegan::procrustes(X, Y))
  }
}
