# Kendall's tau (Reference §4.3)
# From-scratch base R plus stats::cor.test(method = "kendall").
# Run with:  Rscript kendalls_tau.R
#
# Inputs used below:
#   x, y        : paired numeric vectors of equal length
#   alternative : "two.sided" / "less" / "greater"

kendall_counts <- function(x, y) {
  n <- length(x); C <- D <- Tx <- Ty <- Txy <- 0L
  for (i in seq_len(n - 1)) for (j in (i + 1):n) {
    dx <- x[i] - x[j]; dy <- y[i] - y[j]
    if (dx == 0 && dy == 0) Txy <- Txy + 1
    else if (dx == 0) Tx <- Tx + 1
    else if (dy == 0) Ty <- Ty + 1
    else if ((dx > 0) == (dy > 0)) C <- C + 1
    else D <- D + 1
  }
  list(C = C, D = D, Tx = Tx, Ty = Ty, Txy = Txy, n = n)
}

kendall_tau_a_scratch <- function(x, y) {
  k <- kendall_counts(x, y); (k$C - k$D) / (k$n * (k$n - 1) / 2)
}

kendall_tau_b_scratch <- function(x, y) {
  k <- kendall_counts(x, y); n0 <- k$n * (k$n - 1) / 2
  denom <- sqrt((n0 - k$Tx) * (n0 - k$Ty))
  if (denom > 0) (k$C - k$D) / denom else NA_real_
}

kendall_test_scratch <- function(x, y, alternative = "two.sided") {
  k <- kendall_counts(x, y); tau <- kendall_tau_b_scratch(x, y)
  var <- k$n * (k$n - 1) * (2 * k$n + 5) / 18
  z <- (k$C - k$D) / sqrt(var)
  p <- switch(alternative,
    "two.sided" = 2 * pnorm(-abs(z)),
    "greater"   = pnorm(z, lower.tail = FALSE),
    "less"      = pnorm(z))
  list(tau_b = tau, concordant = k$C, discordant = k$D,
       tied_x = k$Tx, tied_y = k$Ty, tied_both = k$Txy,
       z = z, p_value = p, method = "Kendall tau-b", alternative = alternative)
}

# Library: stats::cor(method = "kendall"), stats::cor.test(method = "kendall")
library_demo <- function(x, y) {
  cat("cor (method = 'kendall'):", cor(x, y, method = "kendall"), "\n")
  print(cor.test(x, y, method = "kendall", exact = FALSE))
}

if (sys.nframe() == 0) {
  set.seed(7); n <- 50
  x <- rnorm(n); y <- 0.7 * x + sqrt(1 - 0.49) * rnorm(n)
  cat("=== Kendall on n =", n, " (no ties) ===\n")
  cat("tau-a =", kendall_tau_a_scratch(x, y), "\n")
  print(kendall_test_scratch(x, y))
  cat("\n=== With ties (ordinal 1..5) ===\n")
  set.seed(7); xt <- sample(1:5, 60, replace = TRUE); yt <- sample(1:5, 60, replace = TRUE)
  print(kendall_test_scratch(xt, yt))
  cat("\n--- library ---\n"); library_demo(x, y)
}
