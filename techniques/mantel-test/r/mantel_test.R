# Mantel test (Reference §9.18)
# R via vegan::mantel and vegan::mantel.partial.
# Run with:  Rscript mantel_test.R

if (sys.nframe() == 0) {
  set.seed(11); n <- 40
  loc <- matrix(rnorm(n * 2), n, 2)
  feat <- 0.7 * loc + matrix(rnorm(n * 2, 0, 0.5), n, 2)
  z_val <- matrix(rnorm(n * 2), n, 2)
  Dg <- dist(loc); Df <- dist(feat); Dz <- dist(z_val)
  if (requireNamespace("vegan", quietly = TRUE)) {
    cat("=== vegan::mantel ===\n")
    print(vegan::mantel(Dg, Df, permutations = 999))
    cat("\n=== vegan::mantel.partial (X ~ Y | Z) ===\n")
    print(vegan::mantel.partial(Dg, Df, Dz, permutations = 999))
  }
}
