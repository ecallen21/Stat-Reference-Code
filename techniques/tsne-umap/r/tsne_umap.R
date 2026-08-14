# t-SNE + UMAP (Reference §26.5)
# R via Rtsne::Rtsne and uwot::umap.
# Run with:  Rscript tsne_umap.R

if (sys.nframe() == 0) {
  set.seed(0); n_per <- 100
  X <- rbind(matrix(rnorm(n_per * 20), n_per, 20),
             matrix(rnorm(n_per * 20, 5), n_per, 20),
             matrix(rnorm(n_per * 20, -5), n_per, 20))
  y <- rep(0:2, each = n_per)
  if (requireNamespace("Rtsne", quietly = TRUE)) {
    cat("=== Rtsne::Rtsne ===\n")
    fit <- Rtsne::Rtsne(X, perplexity = 30, verbose = FALSE)
    cat(sprintf("  dims: %s\n", paste(dim(fit$Y), collapse = " x ")))
  }
  if (requireNamespace("uwot", quietly = TRUE)) {
    cat("\n=== uwot::umap ===\n")
    Z <- uwot::umap(X, n_neighbors = 15, min_dist = 0.1)
    cat(sprintf("  dims: %s\n", paste(dim(Z), collapse = " x ")))
  }
}
