# Multidimensional Scaling (Reference §9.32)
# Base R via stats::cmdscale (classical) + MASS::isoMDS (non-metric).
# Run with:  Rscript multidimensional_scaling.R

classical_mds <- function(D, k = 2) {
  D <- as.matrix(D); n <- nrow(D); D2 <- D^2
  H <- diag(n) - matrix(1 / n, n, n)
  B <- -0.5 * H %*% D2 %*% H
  B <- (B + t(B)) / 2
  e <- eigen(B, symmetric = TRUE)
  w <- pmax(e$values[seq_len(k)], 0)
  coords <- e$vectors[, seq_len(k), drop = FALSE] %*% diag(sqrt(w), k, k)
  list(coordinates = coords, eigenvalues = e$values[seq_len(k)])
}

if (sys.nframe() == 0) {
  true_coords <- rbind(c(0, 0), c(3, 0), c(3, 4), c(0, 4), c(1.5, 2))
  D <- as.matrix(dist(true_coords))
  cat("=== Classical MDS ===\n"); print(classical_mds(D, 2))
  cat("\n--- library: stats::cmdscale ---\n"); print(cmdscale(D, k = 2, eig = TRUE))
  if (requireNamespace("MASS", quietly = TRUE)) {
    cat("\n--- library: MASS::isoMDS (non-metric) ---\n")
    print(MASS::isoMDS(D, k = 2))
  }
}
