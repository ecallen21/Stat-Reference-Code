# Breslow-Day test for OR homogeneity across strata (Reference §8.6)
# From-scratch base R plus DescTools::BreslowDayTest (library cross-check).
# Run with:  Rscript breslow_day.R
#
# Input: list of 2x2 matrices, one per stratum.

mh_or <- function(tables) {
  R <- 0; S <- 0
  for (tbl in tables) {
    a <- tbl[1,1]; b <- tbl[1,2]; c <- tbl[2,1]; d <- tbl[2,2]
    N <- a + b + c + d; if (N == 0) next
    R <- R + a * d / N; S <- S + b * c / N
  }
  if (S == 0) Inf else R / S
}

expected_a <- function(n1, m1, N, psi) {
  lo <- max(0, n1 + m1 - N); hi <- min(n1, m1)
  if (psi == 1) return(n1 * m1 / N)
  A <- psi - 1
  B <- -(psi * (n1 + m1) + (N - n1 - m1))
  C <- psi * n1 * m1
  if (abs(A) < 1e-12) return(-C / B)
  disc <- max(0, B^2 - 4 * A * C)
  r1 <- (-B + sqrt(disc)) / (2 * A)
  r2 <- (-B - sqrt(disc)) / (2 * A)
  for (r in c(r1, r2)) if (r >= lo - 1e-9 && r <= hi + 1e-9) return(min(max(r, lo), hi))
  min(max(r1, lo), hi)
}

var_a <- function(E_a, n1, m1, N) {
  parts <- c(E_a, n1 - E_a, m1 - E_a, N - n1 - m1 + E_a)
  if (any(parts <= 0)) return(0)
  1 / sum(1 / parts)
}

breslow_day <- function(tables, tarone = TRUE) {
  psi <- mh_or(tables)
  if (!is.finite(psi) || psi == 0)
    return(list(chi_square = NA, df = max(0, length(tables) - 1), p_value = NA))
  stat <- 0; num <- 0; den <- 0; K_eff <- 0
  for (tbl in tables) {
    a <- tbl[1,1]; b <- tbl[1,2]; c <- tbl[2,1]; d <- tbl[2,2]
    n1 <- a + b; m1 <- a + c; N <- a + b + c + d
    if (N == 0) next
    E <- expected_a(n1, m1, N, psi)
    V <- var_a(E, n1, m1, N)
    if (V == 0) next
    stat <- stat + (a - E)^2 / V
    num <- num + (a - E); den <- den + V
    K_eff <- K_eff + 1
  }
  df <- max(0, K_eff - 1)
  if (tarone && den > 0) {
    stat <- max(0, stat - num^2 / den); method <- "Breslow-Day-Tarone"
  } else method <- "Breslow-Day"
  list(chi_square = stat, df = df,
       p_value = if (df > 0) pchisq(stat, df, lower.tail = FALSE) else NA,
       OR_MH = psi, K_strata = length(tables), method = method)
}

if (sys.nframe() == 0) {
  homo <- list(
    matrix(c(36, 24, 44, 56), 2), matrix(c(30, 22, 50, 58), 2), matrix(c(40, 28, 30, 52), 2)
  )
  cat("=== Homogeneous ===\n"); print(breslow_day(homo, tarone = TRUE))

  hetero <- list(
    matrix(c(40, 10, 20, 30), 2), matrix(c(20, 40, 40, 20), 2), matrix(c(35, 30, 35, 30), 2)
  )
  cat("\n=== Heterogeneous ===\n"); print(breslow_day(hetero, tarone = TRUE))

  if (requireNamespace("DescTools", quietly = TRUE)) {
    cat("\n--- library: DescTools::BreslowDayTest ---\n")
    arr <- array(unlist(homo), dim = c(2, 2, length(homo)))
    print(DescTools::BreslowDayTest(arr, correct = TRUE))
  } else {
    cat("\n(DescTools not installed)\n")
  }
}
