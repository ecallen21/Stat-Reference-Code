# k-means clustering (Reference §9.9)
# Base R via stats::kmeans (with Hartigan-Wong or Lloyd algorithms).
# Run with:  Rscript k_means.R

if (sys.nframe() == 0) {
  set.seed(51)
  centers <- rbind(c(0, 0), c(4, 4), c(8, 0))
  X <- do.call(rbind, lapply(seq_len(nrow(centers)), function(i)
      MASS::mvrnorm(40, centers[i, ], diag(2) * 0.49)))
  km <- kmeans(X, centers = 3, nstart = 10, algorithm = "Lloyd", iter.max = 300)
  cat("=== stats::kmeans (Lloyd, k=3, 10 restarts) ===\n")
  cat("  total within-SS:", km$tot.withinss, "\n")
  cat("  cluster sizes:", km$size, "\n")
  cat("  centroids:\n"); print(km$centers)
}
