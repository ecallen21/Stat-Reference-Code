# PERMANOVA (Reference §9.17)
# R via vegan::adonis2.
# Run with:  Rscript permanova.R

if (sys.nframe() == 0) {
  set.seed(7); n_per <- 30
  X <- rbind(matrix(rnorm(n_per * 4), n_per, 4),
              matrix(rnorm(n_per * 4, mean = 0.7), n_per, 4))
  groups <- factor(rep(c("A", "B"), each = n_per))
  if (requireNamespace("vegan", quietly = TRUE)) {
    cat("=== vegan::adonis2 (PERMANOVA) ===\n")
    print(vegan::adonis2(dist(X) ~ groups, permutations = 999))
  }
}
