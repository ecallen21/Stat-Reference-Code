# Canonical Correlation Analysis (Reference §9.29)
# Base R via stats::cancor + CCA package for library cross-check.
# Run with:  Rscript canonical_correlation.R

canonical_correlation <- function(X, Y) {
  X <- as.matrix(X); Y <- as.matrix(Y); n <- nrow(X); p <- ncol(X); q <- ncol(Y)
  Xc <- scale(X, scale = FALSE); Yc <- scale(Y, scale = FALSE)
  Rxx <- crossprod(Xc) / (n - 1)
  Ryy <- crossprod(Yc) / (n - 1)
  Rxy <- crossprod(Xc, Yc) / (n - 1)
  # Rxx^{-1/2}
  e <- eigen(Rxx, symmetric = TRUE); w <- pmax(e$values, 0)
  Rxx_ih <- e$vectors %*% diag(ifelse(w > 1e-12, w^-0.5, 0)) %*% t(e$vectors)
  Ryy_inv <- MASS::ginv(Ryy)
  M <- Rxx_ih %*% Rxy %*% Ryy_inv %*% t(Rxy) %*% Rxx_ih
  eig <- eigen(M, symmetric = TRUE)
  lam <- pmin(pmax(eig$values, 0), 1)
  ord <- order(-lam); lam <- lam[ord]
  Wx <- Rxx_ih %*% eig$vectors[, ord]
  s <- min(p, q); r <- sqrt(lam[seq_len(s)])
  Wx <- Wx[, seq_len(s), drop = FALSE]
  Wy <- Ryy_inv %*% t(Rxy) %*% Wx
  Wy <- sweep(Wy, 2, ifelse(r > 0, r, 1), "/")
  # Bartlett
  bart <- list()
  for (k in 0:(s - 1)) {
    rem <- r[(k + 1):s]
    wilks <- prod(1 - rem^2)
    st <- -(n - 1 - (p + q + 1) / 2) * log(max(wilks, 1e-300))
    df <- (p - k) * (q - k)
    bart[[k + 1]] <- list(after_k = k, wilks = wilks, chi_square = st, df = df,
                           p_value = pchisq(st, df, lower.tail = FALSE))
  }
  list(canonical_correlations = r,
       X_weights = Wx, Y_weights = Wy,
       bartlett = bart, n = n, p = p, q = q)
}

if (sys.nframe() == 0) {
  set.seed(97); n <- 300
  F <- matrix(rnorm(n * 2), n, 2)
  Lx <- matrix(c(0.8, 0.7, 0.6, 0.1, 0.2, 0.3), 3, 2)
  Ly <- matrix(c(0.1, 0.2, 0.8, 0.7), 2, 2)
  X <- F %*% t(Lx) + matrix(rnorm(n * 3, 0, 0.4), n, 3)
  Y <- F %*% t(Ly) + matrix(rnorm(n * 2, 0, 0.4), n, 2)
  cat("=== From scratch ===\n"); print(canonical_correlation(X, Y))
  cat("\n--- library: stats::cancor ---\n")
  print(cancor(X, Y))
}
