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

# Library: DescTools::GoodmanKruskalGamma, DescTools::SomersDelta,
#          DescTools::KendallTauB (also rcompanion::cliffDelta, etc.)
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
  cat("\n--- library ---\n"); library_demo(table)
}
