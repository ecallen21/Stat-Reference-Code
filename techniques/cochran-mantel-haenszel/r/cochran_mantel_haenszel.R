# Cochran-Mantel-Haenszel test + MH common OR (Reference §8.3, §8.16)
# From-scratch base R plus stats::mantelhaen.test (idiomatic call).
# Run with:  Rscript cochran_mantel_haenszel.R
#
# Input: `tables` -- a list of 2x2 matrices, one per stratum:
#   tbl[1,1] = a (expo+, out+),  tbl[1,2] = b (expo+, out-)
#   tbl[2,1] = c (expo-, out+),  tbl[2,2] = d (expo-, out-)

cmh_test <- function(tables, continuity = FALSE) {
  numer <- 0; denom <- 0
  for (tbl in tables) {
    a <- tbl[1, 1]; b <- tbl[1, 2]; c <- tbl[2, 1]; d <- tbl[2, 2]
    N <- a + b + c + d; if (N <= 1) next
    n1 <- a + b; n0 <- c + d; m1 <- a + c; m0 <- b + d
    e_a <- n1 * m1 / N
    var_a <- (n1 * n0 * m1 * m0) / (N^2 * (N - 1))
    numer <- numer + (a - e_a); denom <- denom + var_a
  }
  if (denom == 0) return(list(chi_square = 0, df = 1, p_value = 1))
  num2 <- max(0, abs(numer) - if (continuity) 0.5 else 0)
  x2 <- num2^2 / denom
  list(chi_square = x2, df = 1,
       p_value = pchisq(x2, 1, lower.tail = FALSE),
       continuity = continuity)
}

mh_common_or <- function(tables) {
  R <- 0; S <- 0; sumPR <- 0; sumPSQR <- 0; sumQS <- 0
  for (tbl in tables) {
    a <- tbl[1, 1]; b <- tbl[1, 2]; c <- tbl[2, 1]; d <- tbl[2, 2]
    N <- a + b + c + d; if (N == 0) next
    Rk <- a * d / N; Sk <- b * c / N
    Pk <- (a + d) / N; Qk <- (b + c) / N
    R <- R + Rk; S <- S + Sk
    sumPR <- sumPR + Pk * Rk
    sumPSQR <- sumPSQR + Pk * Sk + Qk * Rk
    sumQS <- sumQS + Qk * Sk
  }
  or_mh <- R / S
  var_log <- sumPR / (2 * R^2) + sumPSQR / (2 * R * S) + sumQS / (2 * S^2)
  se <- sqrt(var_log); z <- qnorm(0.975)
  list(OR_MH = or_mh, log_OR_SE = se,
       CI95_lower = exp(log(or_mh) - z * se),
       CI95_upper = exp(log(or_mh) + z * se))
}

woolf_homogeneity <- function(tables) {
  log_ors <- numeric(length(tables))
  weights <- numeric(length(tables))
  for (i in seq_along(tables)) {
    tbl <- tables[[i]]
    a <- tbl[1, 1]; b <- tbl[1, 2]; c <- tbl[2, 1]; d <- tbl[2, 2]
    if (any(c(a, b, c, d) == 0)) { a <- a + 0.5; b <- b + 0.5; c <- c + 0.5; d <- d + 0.5 }
    log_ors[i] <- log(a * d / (b * c))
    weights[i] <- 1 / (1 / a + 1 / b + 1 / c + 1 / d)
  }
  lo_bar <- sum(weights * log_ors) / sum(weights)
  x2 <- sum(weights * (log_ors - lo_bar)^2)
  K <- length(tables)
  list(chi_square = x2, df = K - 1,
       p_value = pchisq(x2, K - 1, lower.tail = FALSE),
       OR_pooled = exp(lo_bar))
}

if (sys.nframe() == 0) {
  tables <- list(
    matrix(c(36, 24, 44, 56), nrow = 2, byrow = FALSE),  # stratum 1
    matrix(c(30, 22, 50, 58), nrow = 2, byrow = FALSE),  # stratum 2
    matrix(c(40, 28, 30, 52), nrow = 2, byrow = FALSE)   # stratum 3
  )
  cat("=== CMH ===\n"); print(cmh_test(tables))
  cat("\n=== CMH continuity ===\n"); print(cmh_test(tables, continuity = TRUE))
  cat("\n=== MH common OR + RBG CI ===\n"); print(mh_common_or(tables))
  cat("\n=== Woolf homogeneity ===\n"); print(woolf_homogeneity(tables))

  # Reshape list-of-matrices -> 3D array [outcome, exposure, stratum] for R's built-in
  arr <- array(unlist(tables), dim = c(2, 2, length(tables)))
  cat("\n--- library: stats::mantelhaen.test ---\n")
  print(mantelhaen.test(arr, correct = FALSE))
}
