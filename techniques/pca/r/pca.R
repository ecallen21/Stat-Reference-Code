# Principal Component Analysis (Reference §9.3)
# From-scratch base R via SVD + stats::prcomp as library cross-check.
# Run with:  Rscript pca.R
#
# Inputs:
#   X : n x p numeric matrix
#   scale : TRUE for correlation-matrix PCA (unit-SD variables); FALSE for covariance

pca_scratch <- function(X, n_components = NULL, scale = FALSE) {
  X <- as.matrix(X); n <- nrow(X); p <- ncol(X)
  if (is.null(n_components) || n_components > min(n, p)) n_components <- min(n, p)
  center <- colMeans(X)
  scl <- if (scale) apply(X, 2, sd) else rep(1, p)
  Xc <- sweep(sweep(X, 2, center), 2, ifelse(scl > 0, scl, 1), "/")
  sv <- svd(Xc / sqrt(n - 1))
  V <- sv$v[, seq_len(n_components), drop = FALSE]
  d <- sv$d[seq_len(n_components)]
  scores <- Xc %*% V
  variances <- d^2
  total_var <- sum(apply(Xc, 2, var))
  ratio <- variances / total_var
  list(n_components = n_components, center = center, scale = if (scale) scl else NULL,
       loadings = V, scores = scores,
       singular_values = d, explained_variance = variances,
       explained_variance_ratio = ratio,
       cumulative_variance_ratio = cumsum(ratio))
}

if (sys.nframe() == 0) {
  set.seed(23); n <- 200
  s <- cbind(rnorm(n, 0, 3), rnorm(n, 0, 1.5))
  L <- matrix(c(0.6, 0.5, 0.4, 0.3, 0.2,
                0.1,-0.2, 0.3,-0.5, 0.7), nrow = 5)
  L <- sweep(L, 2, sqrt(colSums(L^2)), "/")
  X <- s %*% t(L) + matrix(rnorm(n * 5, 0, 0.3), n, 5) +
       matrix(1:5, n, 5, byrow = TRUE)
  cat("=== From-scratch PCA ===\n"); print(pca_scratch(X, 5))
  cat("\n--- library: prcomp ---\n")
  print(prcomp(X, scale. = FALSE))
  print(summary(prcomp(X)))
}
