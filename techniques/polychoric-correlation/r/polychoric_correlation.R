# Tetrachoric and polychoric correlation (Reference §4.7)
# From-scratch base R plus polycor::polychor / psych::polychoric.
# Run with:  Rscript polychoric_correlation.R
#
# Inputs used below:
#   table : r x c matrix of nonnegative integer counts (ordinal-by-ordinal)

thresholds_from_marginals <- function(margins) {
  cum <- cumsum(margins); cum <- cum[-length(cum)]
  qnorm(pmin(pmax(cum, 1e-8), 1 - 1e-8))
}

cell_prob <- function(a_lo, a_hi, b_lo, b_hi, rho) {
  bvn <- function(a, b) mvtnorm::pmvnorm(upper = c(a, b),
                                          corr = matrix(c(1, rho, rho, 1), 2))
  if (!requireNamespace("mvtnorm", quietly = TRUE))
    stop("install 'mvtnorm' for bivariate normal CDF")
  a_hi <- if (is.infinite(a_hi)) 8 else a_hi
  a_lo <- if (is.infinite(a_lo)) -8 else a_lo
  b_hi <- if (is.infinite(b_hi)) 8 else b_hi
  b_lo <- if (is.infinite(b_lo)) -8 else b_lo
  as.numeric(bvn(a_hi, b_hi) - bvn(a_lo, b_hi) - bvn(a_hi, b_lo) + bvn(a_lo, b_lo))
}

polychoric_scratch <- function(table) {
  obs <- as.matrix(table); r <- nrow(obs); c <- ncol(obs); n <- sum(obs)
  row_m <- rowSums(obs) / n; col_m <- colSums(obs) / n
  alpha <- c(-Inf, thresholds_from_marginals(row_m), Inf)
  beta  <- c(-Inf, thresholds_from_marginals(col_m), Inf)
  neg_ll <- function(rho) {
    ll <- 0
    for (i in seq_len(r)) for (j in seq_len(c)) {
      if (obs[i, j] == 0) next
      p <- cell_prob(alpha[i], alpha[i + 1], beta[j], beta[j + 1], rho)
      ll <- ll + obs[i, j] * log(max(p, 1e-15))
    }
    -ll
  }
  res <- optimize(neg_ll, c(-0.999, 0.999), tol = 1e-6)
  list(rho = res$minimum, row_thresholds = alpha[2:(r)], col_thresholds = beta[2:(c)],
       log_likelihood = -res$objective, n = n, size = c(r, c),
       method = if (r == 2 && c == 2) "tetrachoric" else "polychoric")
}

# Library: polycor::polychor (or polycor::polychor for 2x2 = tetrachoric),
#          psych::polychoric (also returns SEs and a correlation matrix)
library_demo <- function(table) {
  if (requireNamespace("polycor", quietly = TRUE)) {
    cat("polycor::polychor:\n"); print(polycor::polychor(table, std.err = TRUE))
  } else cat("(install 'polycor' for polychor)\n")
}

if (sys.nframe() == 0) {
  table <- matrix(c(20, 8, 2, 12, 30, 10, 3, 12, 25), nrow = 3, byrow = TRUE)
  cat("=== Polychoric (3x3) ===\n");  print(polychoric_scratch(table))
  cat("\n=== Tetrachoric (2x2) ===\n")
  print(polychoric_scratch(matrix(c(60, 15, 10, 40), nrow = 2, byrow = TRUE)))
  cat("\n--- library ---\n"); library_demo(table)
}
