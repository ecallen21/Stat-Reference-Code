# Cramer's V and the phi coefficient (Reference §4.10)
# From-scratch base R plus DescTools::CramerV / vcd::assocstats.
# Run with:  Rscript cramers_v_phi.R
#
# Inputs used below:
#   table         : r x c contingency table (matrix of counts)
#   bias_correct  : apply the Bergsma (2013) bias correction

chi_square_scratch <- function(table) {
  obs <- as.matrix(table); n <- sum(obs)
  expected <- outer(rowSums(obs), colSums(obs)) / n
  sum(((obs - expected)^2 / expected)[expected > 0])
}

phi_coefficient_scratch <- function(table) {
  obs <- as.matrix(table)
  if (!all(dim(obs) == c(2, 2))) return(NA_real_)
  a <- obs[1, 1]; b <- obs[1, 2]; c <- obs[2, 1]; d <- obs[2, 2]
  denom <- sqrt((a + b) * (c + d) * (a + c) * (b + d))
  if (denom > 0) (a * d - b * c) / denom else NA_real_
}

cramers_v_scratch <- function(table, bias_correct = FALSE) {
  obs <- as.matrix(table); r <- nrow(obs); c <- ncol(obs); n <- sum(obs)
  chi2 <- chi_square_scratch(obs); phi2 <- chi2 / n
  V <- sqrt(phi2 / min(r - 1, c - 1))
  out <- list(chi_square = chi2, n = n, shape = c(r, c),
              phi_squared = phi2, cramers_v = V)
  if (bias_correct) {
    r_tilde <- r - (r - 1)^2 / (n - 1); c_tilde <- c - (c - 1)^2 / (n - 1)
    phi2_corr <- max(0, phi2 - (r - 1) * (c - 1) / (n - 1))
    denom <- min(r_tilde - 1, c_tilde - 1)
    out$cramers_v_bias_corrected <- if (denom > 0) sqrt(phi2_corr / denom) else 0
  }
  if (r == 2 && c == 2) out$phi_signed <- phi_coefficient_scratch(obs)
  out
}

# Library: DescTools::CramerV, vcd::assocstats, rstatix::cramer_v
library_demo <- function(table) {
  if (requireNamespace("DescTools", quietly = TRUE)) {
    cat("DescTools::CramerV:", DescTools::CramerV(table), "\n")
    cat("DescTools::CramerV bias-corrected:", DescTools::CramerV(table, conf.level = 0.95, method = "ncchisqadj"), "\n")
  } else cat("(install 'DescTools' for CramerV)\n")
  if (requireNamespace("vcd", quietly = TRUE)) {
    cat("\nvcd::assocstats:\n"); print(vcd::assocstats(table))
  }
}

if (sys.nframe() == 0) {
  cat("=== 2x2 ===\n");  print(cramers_v_scratch(matrix(c(40, 20, 10, 30), 2), bias_correct = TRUE))
  cat("\n=== 3x4 ===\n")
  tbl <- matrix(c(30, 10, 5,  20, 25, 10,  10, 30, 25,  5, 15, 30),
                nrow = 3, byrow = TRUE)
  print(cramers_v_scratch(tbl, bias_correct = TRUE))
  cat("\n--- library ---\n"); library_demo(tbl)
}
