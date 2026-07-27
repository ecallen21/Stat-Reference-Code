# Cluster validation indices (Reference §9.14)
# From-scratch base R + cluster::silhouette + fpc::cluster.stats + cluster::clusGap.
# Run with:  Rscript cluster_validation.R

silhouette_scratch <- function(X, labels) {
  D <- as.matrix(dist(X)); n <- nrow(X); uc <- unique(labels)
  per <- numeric(n)
  for (i in seq_len(n)) {
    own <- labels[i]
    same <- labels == own; same[i] <- FALSE
    if (!any(same)) { per[i] <- 0; next }
    a <- mean(D[i, same])
    b <- Inf
    for (c in uc) if (c != own) b <- min(b, mean(D[i, labels == c]))
    per[i] <- (b - a) / max(a, b)
  }
  list(overall = mean(per), per_cluster = tapply(per, labels, mean))
}

calinski_harabasz <- function(X, labels) {
  X <- as.matrix(X); n <- nrow(X); p <- ncol(X)
  uc <- unique(labels); k <- length(uc)
  if (k < 2) return(NA)
  grand <- colMeans(X); B <- 0; W <- 0
  for (c in uc) {
    Xc <- X[labels == c, , drop = FALSE]; nc <- nrow(Xc); cent <- colMeans(Xc)
    B <- B + nc * sum((cent - grand)^2)
    W <- W + sum((sweep(Xc, 2, cent))^2)
  }
  (B / (k - 1)) / (W / (n - k))
}

davies_bouldin <- function(X, labels) {
  X <- as.matrix(X); uc <- unique(labels); k <- length(uc)
  cents <- t(sapply(uc, function(c) colMeans(X[labels == c, , drop = FALSE])))
  disp <- sapply(seq_len(k), function(i) mean(sqrt(rowSums(sweep(
      X[labels == uc[i], , drop = FALSE], 2, cents[i, ])^2))))
  d <- as.matrix(dist(cents)); diag(d) <- Inf
  mean(apply(outer(disp, disp, "+") / d, 1, max))
}

if (sys.nframe() == 0) {
  set.seed(83)
  centers <- rbind(c(0, 0), c(4, 4), c(8, 0))
  X <- do.call(rbind, lapply(seq_len(nrow(centers)), function(i)
      MASS::mvrnorm(50, centers[i, ], diag(2) * 0.49)))
  km <- kmeans(X, 3, nstart = 10)
  cat("=== validation at k=3 ===\n")
  print(silhouette_scratch(X, km$cluster))
  cat("CH:", calinski_harabasz(X, km$cluster), "\n")
  cat("DB:", davies_bouldin(X, km$cluster), "\n")

  if (requireNamespace("cluster", quietly = TRUE)) {
    cat("\n--- library: cluster::silhouette + clusGap ---\n")
    print(summary(cluster::silhouette(km$cluster, dist(X))))
    print(cluster::clusGap(X, kmeans, K.max = 7, B = 20, nstart = 10))
  }
}
