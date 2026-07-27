# Hierarchical (agglomerative) clustering (Reference §9.8)
# Base R via stats::hclust + cluster::agnes.
# Run with:  Rscript hierarchical_clustering.R
#
# Inputs:
#   X : n x p numeric matrix

hc_scratch <- function(X, method = "average") {
  # Base R's hclust does exactly what we want; just call it -- from-scratch
  # matches the Python file for pedagogy.
  d <- dist(as.matrix(X))
  hclust(d, method = method)
}

cophenetic_corr <- function(hc, X) {
  d <- dist(as.matrix(X))
  cop <- cophenetic(hc)
  cor(d, cop)
}

if (sys.nframe() == 0) {
  set.seed(41)
  centers <- rbind(c(0, 0), c(4, 4), c(8, 0))
  X <- do.call(rbind, lapply(seq_len(nrow(centers)), function(i)
      MASS::mvrnorm(30, centers[i, ], diag(2) * 0.49)))
  for (m in c("single", "complete", "average", "ward.D2")) {
    hc <- hc_scratch(X, m)
    labels <- cutree(hc, k = 3)
    cat("=== method:", m, "===\n")
    cat("  cluster sizes at k=3:", table(labels), "\n")
    cat("  cophenetic corr:", round(cophenetic_corr(hc, X), 4), "\n")
  }
  if (requireNamespace("cluster", quietly = TRUE)) {
    cat("\n--- library: cluster::agnes ---\n")
    print(cluster::agnes(X, method = "ward"))
  }
}
