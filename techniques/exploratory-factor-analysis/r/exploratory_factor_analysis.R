# Exploratory Factor Analysis (Reference §9.4)
# From-scratch base R (PAF + varimax + promax) + psych::fa / stats::factanal
# as library cross-checks.
# Run with:  Rscript exploratory_factor_analysis.R
#
# Input:
#   X : n x p numeric matrix of item scores

paf <- function(R, n_factors, max_iter = 100, tol = 1e-6) {
  p <- ncol(R)
  R_inv <- MASS::ginv(R)
  h2 <- pmax(0.05, pmin(0.99, 1 - 1 / pmax(diag(R_inv), 1e-12)))
  L <- NULL
  for (it in seq_len(max_iter)) {
    R_red <- R; diag(R_red) <- h2
    e <- eigen(R_red, symmetric = TRUE)
    w <- pmax(e$values[seq_len(n_factors)], 1e-12)
    v <- e$vectors[, seq_len(n_factors), drop = FALSE]
    L <- sweep(v, 2, sqrt(w), "*")
    h2_new <- pmax(0.05, pmin(0.99, rowSums(L^2)))
    if (max(abs(h2_new - h2)) < tol) { h2 <- h2_new; break }
    h2 <- h2_new
  }
  list(loadings = L, communalities = h2)
}

varimax_scratch <- function(L, gamma = 1, max_iter = 100, tol = 1e-8) {
  p <- nrow(L); k <- ncol(L)
  R <- diag(k); d <- 0
  for (it in seq_len(max_iter)) {
    d_old <- d
    Lam <- L %*% R
    B <- t(L) %*% (Lam^3 - (gamma / p) * Lam %*% diag(colSums(Lam^2)))
    sv <- svd(B)
    R <- sv$u %*% t(sv$v)
    d <- sum(sv$d)
    if (abs(d - d_old) < tol) break
  }
  L %*% R
}

promax_scratch <- function(L, kappa = 4) {
  Lv <- varimax_scratch(L)
  target <- sign(Lv) * abs(Lv)^kappa
  T <- solve(t(Lv) %*% Lv, t(Lv) %*% target)
  A_inv <- solve(t(T) %*% T)
  D <- diag(sqrt(diag(A_inv)))
  T <- T %*% D
  Phi <- solve(t(T) %*% T)
  list(loadings = Lv %*% T, Phi = Phi)
}

fit_efa <- function(X, n_factors, rotation = "varimax") {
  X <- as.matrix(X); n <- nrow(X); p <- ncol(X)
  R <- cor(X)
  pa <- paf(R, n_factors)
  L <- pa$loadings; Phi <- NULL
  if (rotation == "varimax")   L <- varimax_scratch(L)
  else if (rotation == "promax") {
    pr <- promax_scratch(L); L <- pr$loadings; Phi <- pr$Phi
  }
  if (is.null(Phi)) h2 <- rowSums(L^2) else h2 <- diag(L %*% Phi %*% t(L))
  list(loadings = L, communalities = h2, uniquenesses = 1 - h2,
       ss_loadings = colSums(L^2), Phi = Phi,
       n = n, p = p, k = n_factors, rotation = rotation)
}

if (sys.nframe() == 0) {
  set.seed(31); n <- 300
  F <- matrix(rnorm(n * 2), n, 2)
  L_true <- matrix(c(0.8, 0.7, 0.9, 0.1, 0.2, 0.0,
                     0.1, 0.2, 0.0, 0.7, 0.8, 0.9),
                    nrow = 6)
  U <- matrix(rnorm(n * 6, 0, 0.5), n, 6)
  X <- F %*% t(L_true) + U
  cat("=== PAF + varimax, k = 2 ===\n"); print(fit_efa(X, 2, "varimax"))
  cat("\n=== PAF + promax, k = 2 ===\n"); print(fit_efa(X, 2, "promax"))
  if (requireNamespace("psych", quietly = TRUE)) {
    cat("\n--- library: psych::fa ---\n")
    print(psych::fa(X, nfactors = 2, rotate = "varimax", fm = "pa"))
  }
}
