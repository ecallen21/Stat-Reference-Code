# ANOSIM (Reference §9.19)
# R via vegan::anosim.
# Run with:  Rscript anosim.R

if (sys.nframe() == 0) {
  set.seed(4); n_per <- 20
  X <- rbind(matrix(rnorm(n_per * 3, mean =  0.0), n_per, 3),
             matrix(rnorm(n_per * 3, mean =  1.2), n_per, 3),
             matrix(rnorm(n_per * 3, mean = -1.2), n_per, 3))
  groups <- factor(rep(c("A", "B", "C"), each = n_per))
  D <- dist(X)
  if (requireNamespace("vegan", quietly = TRUE)) {
    cat("=== vegan::anosim (3 groups, shifted centers) ===\n")
    print(vegan::anosim(D, grouping = groups, permutations = 999))
  }
}
