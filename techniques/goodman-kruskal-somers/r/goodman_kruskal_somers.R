# Goodman-Kruskal gamma, Somers' D, Kendall's tau-b on contingency tables
# (Reference §4.11, §4.12)
# From-scratch base R plus DescTools::GoodmanKruskalGamma / DescTools::SomersDelta.
# Run with:  Rscript goodman_kruskal_somers.R
#
# Inputs used below:
#   table : r x c contingency table of counts (rows = ordinal X, cols = ordinal Y)

ordinal_pair_counts <- function(table) {
  obs <- as.matrix(table); r <- nrow(obs); c <- ncol(obs); C <- D <- Tr <- Tc <- 0
  for (i in seq_len(r)) for (j in seq_len(c)) {
    nij <- obs[i, j]; if (nij == 0) next
    if (i < r && j < c)
      C <- C + nij * sum(obs[(i + 1):r, (j + 1):c])
    if (i < r && j > 1)
      D <- D + nij * sum(obs[(i + 1):r, 1:(j - 1)])
  }
  for (i in seq_len(r)) for (j in seq_len(c - 1)) for (j2 in (j + 1):c)
    Tr <- Tr + obs[i, j] * obs[i, j2]
  for (j in seq_len(c)) for (i in seq_len(r - 1)) for (i2 in (i + 1):r)
    Tc <- Tc + obs[i, j] * obs[i2, j]
  list(C = C, D = D, T_x_only = Tr, T_y_only = Tc)
}

goodman_kruskal_gamma_scratch <- function(table) {
  k <- ordinal_pair_counts(table); if ((k$C + k$D) == 0) NA else (k$C - k$D) / (k$C + k$D)
}

somers_d_y_given_x <- function(table) {
  k <- ordinal_pair_counts(table); denom <- k$C + k$D + k$T_y_only
  if (denom > 0) (k$C - k$D) / denom else NA
}

somers_d_x_given_y <- function(table) somers_d_y_given_x(t(table))

kendall_tau_b_table <- function(table) {
  k <- ordinal_pair_counts(table); denom <- sqrt((k$C + k$D + k$T_x_only) * (k$C + k$D + k$T_y_only))
  if (denom > 0) (k$C - k$D) / denom else NA
}

all_ordinal_associations <- function(table) {
  k <- ordinal_pair_counts(table)
  list(C = k$C, D = k$D, T_x_only = k$T_x_only, T_y_only = k$T_y_only,
       gamma = goodman_kruskal_gamma_scratch(table),
       somers_d_y_given_x = somers_d_y_given_x(table),
       somers_d_x_given_y = somers_d_x_given_y(table),
       kendall_tau_b = kendall_tau_b_table(table))
}

# --- Goodman-Kruskal nominal PRE measures ---------------------------------
# lambda(y|x): PRE for predicting Y from X via the mode rule
gk_lambda <- function(table, predict = "y_given_x") {
  obs <- as.matrix(table)
  if (predict == "x_given_y") obs <- t(obs)
  else if (predict != "y_given_x") stop("predict must be 'y_given_x' or 'x_given_y'")
  N <- sum(obs); col_tot <- colSums(obs)
  P_e <- 1 - max(col_tot) / N
  P_e_given_x <- (N - sum(apply(obs, 1, max))) / N
  if (P_e > 0) (P_e - P_e_given_x) / P_e else NA_real_
}

# tau(y|x): variance-based PRE; "Goodman-Kruskal tau" is NOT Kendall's tau
gk_tau <- function(table, predict = "y_given_x") {
  obs <- as.matrix(table)
  if (predict == "x_given_y") obs <- t(obs)
  else if (predict != "y_given_x") stop("predict must be 'y_given_x' or 'x_given_y'")
  N <- sum(obs); row_tot <- rowSums(obs); col_tot <- colSums(obs)
  V_y <- 1 - sum((col_tot / N)^2)
  V_y_given_x <- 0
  for (i in seq_along(row_tot)) {
    if (row_tot[i] == 0) next
    V_y_given_x <- V_y_given_x + (row_tot[i] / N) * (1 - sum((obs[i, ] / row_tot[i])^2))
  }
  if (V_y > 0) (V_y - V_y_given_x) / V_y else NA_real_
}

# Library: DescTools::GoodmanKruskalGamma, DescTools::SomersDelta,
#          DescTools::KendallTauB, DescTools::Lambda, DescTools::GoodmanKruskalTau
library_demo <- function(table) {
  if (requireNamespace("DescTools", quietly = TRUE)) {
    cat("DescTools::GoodmanKruskalGamma:", DescTools::GoodmanKruskalGamma(table), "\n")
    cat("DescTools::SomersDelta (D(y|x)):", DescTools::SomersDelta(table, direction = "column"), "\n")
    cat("DescTools::KendallTauB:", DescTools::KendallTauB(table), "\n")
  } else cat("(install 'DescTools' for these measures)\n")
}

if (sys.nframe() == 0) {
  table <- matrix(c(40, 20,  5,  15, 30, 20,  5, 15, 50), nrow = 3, byrow = TRUE)
  cat("=== Ordinal-by-ordinal table ===\n"); print(table)
  print(all_ordinal_associations(table))
  cat("\n=== Nominal PRE measures on the same table ===\n")
  cat(sprintf("  G-K lambda (y|x): %.4f\n", gk_lambda(table, "y_given_x")))
  cat(sprintf("  G-K lambda (x|y): %.4f\n", gk_lambda(table, "x_given_y")))
  cat(sprintf("  G-K tau    (y|x): %.4f\n", gk_tau(table, "y_given_x")))
  cat(sprintf("  G-K tau    (x|y): %.4f\n", gk_tau(table, "x_given_y")))
  cat("\n--- library ---\n"); library_demo(table)
}
