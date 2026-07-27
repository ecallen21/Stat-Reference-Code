# Correspondence Analysis (Reference §8.5)
# From-scratch base R via SVD, plus MASS::corresp / ca::ca cross-checks.
# Run with:  Rscript correspondence_analysis.R

correspondence_analysis <- function(counts, row_labels = NULL, col_labels = NULL,
                                     n_components = NULL) {
  N <- as.matrix(counts); I <- nrow(N); J <- ncol(N)
  if (is.null(row_labels)) row_labels <- rownames(N)
  if (is.null(col_labels)) col_labels <- colnames(N)
  n <- sum(N); P <- N / n
  r <- rowSums(P); c <- colSums(P)
  Dr_ih <- diag(1 / sqrt(pmax(r, 1e-12)))
  Dc_ih <- diag(1 / sqrt(pmax(c, 1e-12)))
  S <- Dr_ih %*% (P - outer(r, c)) %*% Dc_ih
  sv <- svd(S)
  keep <- sv$d > 1e-12
  U <- sv$u[, keep, drop = FALSE]; V <- sv$v[, keep, drop = FALSE]; d <- sv$d[keep]
  max_dim <- min(I, J) - 1
  if (is.null(n_components)) n_components <- max_dim
  n_components <- min(n_components, max_dim)
  U <- U[, seq_len(n_components), drop = FALSE]
  V <- V[, seq_len(n_components), drop = FALSE]
  d <- d[seq_len(n_components)]
  F <- Dr_ih %*% U %*% diag(d, n_components, n_components)
  G <- Dc_ih %*% V %*% diag(d, n_components, n_components)
  total_inertia <- sum(d^2)
  list(n_components = n_components,
       singular_values = d,
       eigenvalues = d^2,
       total_inertia = total_inertia,
       explained_pct = 100 * d^2 / total_inertia,
       chi_square_total = total_inertia * n,
       df_total = (I - 1) * (J - 1),
       row_coords_principal = F,
       col_coords_principal = G,
       row_labels = row_labels, col_labels = col_labels)
}

if (sys.nframe() == 0) {
  counts <- matrix(c(15,  2,  5,
                      4, 20,  3,
                      3,  3,  8),
                    nrow = 3, byrow = TRUE,
                    dimnames = list(hair = c("blond", "brown", "black"),
                                    eye = c("blue", "brown", "green")))
  cat("=== CA ===\n")
  ca <- correspondence_analysis(counts)
  cat("chi^2 =", ca$chi_square_total, "df =", ca$df_total, "\n")
  cat("total inertia =", ca$total_inertia, "\n")
  cat("singular values =", ca$singular_values, "\n")
  cat("explained inertia % =", ca$explained_pct, "\n")
  cat("row principal coords:\n"); print(ca$row_coords_principal)
  cat("col principal coords:\n"); print(ca$col_coords_principal)

  if (requireNamespace("MASS", quietly = TRUE)) {
    cat("\n--- library: MASS::corresp ---\n")
    print(MASS::corresp(counts, nf = 2))
  }
}
