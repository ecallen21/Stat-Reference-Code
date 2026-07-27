# DBSCAN density-based clustering (Reference §9.11)
# From-scratch base R + dbscan::dbscan as library cross-check.
# Run with:  Rscript dbscan.R

neighborhood <- function(X, i, eps) {
  d <- sqrt(rowSums(sweep(X, 2, X[i, ], "-")^2))
  which(d <= eps)
}

dbscan_scratch <- function(X, eps, min_pts) {
  X <- as.matrix(X); n <- nrow(X)
  labels <- rep(-1L, n); visited <- rep(FALSE, n); is_core <- rep(FALSE, n)
  cluster_id <- 0L
  for (i in seq_len(n)) {
    if (visited[i]) next
    visited[i] <- TRUE
    nbrs <- neighborhood(X, i, eps)
    if (length(nbrs) < min_pts) next     # stays noise (may become border below)
    labels[i] <- cluster_id; is_core[i] <- TRUE
    seeds <- as.list(nbrs); k <- 1L
    while (k <= length(seeds)) {
      j <- seeds[[k]]
      if (!visited[j]) {
        visited[j] <- TRUE
        jn <- neighborhood(X, j, eps)
        if (length(jn) >= min_pts) {
          is_core[j] <- TRUE
          new_ones <- setdiff(jn, unlist(seeds))
          seeds <- c(seeds, as.list(new_ones))
        }
      }
      if (labels[j] == -1L) labels[j] <- cluster_id
      k <- k + 1L
    }
    cluster_id <- cluster_id + 1L
  }
  list(labels = labels, n_clusters = cluster_id,
       n_noise = sum(labels == -1L),
       n_core = sum(is_core),
       cluster_sizes = as.integer(table(labels[labels >= 0])))
}

k_distance <- function(X, k) {
  d <- as.matrix(dist(X))
  diag(d) <- Inf
  sort(apply(d, 1, function(row) sort(row)[k]))
}

if (sys.nframe() == 0) {
  set.seed(61); n_per <- 100
  t1 <- runif(n_per, 0, pi)
  X1 <- cbind(cos(t1) + rnorm(n_per, 0, 0.08),
              sin(t1) + rnorm(n_per, 0, 0.08))
  t2 <- runif(n_per, 0, pi)
  X2 <- cbind(1 + cos(t2) + rnorm(n_per, 0, 0.08),
              -sin(t2) + 0.5 + rnorm(n_per, 0, 0.08))
  noise <- matrix(runif(40, -1, 2), 20, 2)
  X <- rbind(X1, X2, noise)
  cat("=== DBSCAN (eps=0.15, min_pts=4) ===\n")
  print(dbscan_scratch(X, 0.15, 4)[c("n_clusters", "n_noise", "n_core", "cluster_sizes")])
  if (requireNamespace("dbscan", quietly = TRUE)) {
    cat("\n--- library: dbscan::dbscan ---\n")
    print(dbscan::dbscan(X, eps = 0.15, minPts = 4))
  }
}
