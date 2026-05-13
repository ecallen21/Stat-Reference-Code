# Formal tests for univariate outliers (Reference §3.25)
# From-scratch base R plus outliers::grubbs.test / outliers::dixon.test.
# Run with:  Rscript outlier_tests.R
#
# Inputs used below:
#   x           : numeric sample (assumed normal under H0)
#   alpha       : significance level
#   alternative : "two.sided" / "max" / "min"  (Grubbs)
#   r           : maximum number of outliers to look for (Generalized ESD)
#   k           : IQR fence multiplier (1.5 = standard outlier, 3 = "far out")

grubbs_scratch <- function(x, alpha = 0.05, alternative = "two.sided") {
  n <- length(x); m <- mean(x); s <- sd(x); if (s == 0) return(NA)
  devs <- (x - m) / s
  i <- switch(alternative,
    "max" = which.max(devs), "min" = which.min(devs),
    "two.sided" = which.max(abs(devs)))
  G <- if (alternative == "min") -devs[i] else if (alternative == "max") devs[i] else abs(devs[i])
  p_t <- if (alternative == "two.sided") alpha / (2 * n) else alpha / n
  tc <- qt(1 - p_t, n - 2)
  G_crit <- ((n - 1) / sqrt(n)) * sqrt(tc^2 / (n - 2 + tc^2))
  t_obs <- sqrt((n - 2) * G^2 / max(1e-300, (n - 1)^2 / n - G^2))
  p_val <- min(1, pt(t_obs, n - 2, lower.tail = FALSE) *
                  if (alternative == "two.sided") 2 * n else n)
  list(G = unname(G), G_critical = G_crit, p_value = p_val,
       candidate_index = unname(i), candidate_value = unname(x[i]),
       reject_normal = G > G_crit, alternative = alternative)
}

generalized_esd <- function(x, r = 3, alpha = 0.05) {
  data <- x; indices <- seq_along(x); R <- crit <- removed_idx <- removed_val <- numeric()
  for (i in seq_len(r)) {
    n <- length(data); if (n <= 2) break
    m <- mean(data); s <- sd(data); if (s == 0) break
    devs <- (data - m) / s
    j <- which.max(abs(devs)); Ri <- abs(devs[j])
    p <- 1 - alpha / (2 * (n - i + 1))
    tp <- qt(p, n - i - 1)
    lam <- ((n - i) * tp) / sqrt((n - i - 1 + tp^2) * (n - i + 1))
    R <- c(R, Ri); crit <- c(crit, lam)
    removed_idx <- c(removed_idx, indices[j]); removed_val <- c(removed_val, data[j])
    data <- data[-j]; indices <- indices[-j]
  }
  L <- 0
  for (k in length(R):1) if (R[k] > crit[k]) { L <- k; break }
  list(R = R, critical = crit, L_detected = L,
       outlier_indices = removed_idx[seq_len(L)],
       outlier_values = removed_val[seq_len(L)])
}

DIXON_Q_CRIT <- list(
  `3` = c(0.941, 0.970, 0.994), `4` = c(0.765, 0.829, 0.926),
  `5` = c(0.642, 0.710, 0.821), `6` = c(0.560, 0.625, 0.740),
  `7` = c(0.507, 0.568, 0.680), `8` = c(0.468, 0.526, 0.634),
  `9` = c(0.437, 0.493, 0.598), `10` = c(0.412, 0.466, 0.568))

dixons_q_scratch <- function(x, alpha = 0.05) {
  n <- length(x); stopifnot(n >= 3, n <= 10)
  s <- sort(x); rng <- s[n] - s[1]
  if (rng == 0) return(NA)
  Q_max <- (s[n] - s[n - 1]) / rng
  Q_min <- (s[2] - s[1]) / rng
  if (Q_max >= Q_min) { Q <- Q_max; cand <- s[n] } else { Q <- Q_min; cand <- s[1] }
  col <- c(`0.1` = 1, `0.05` = 2, `0.01` = 3)[as.character(alpha)]
  Q_crit <- DIXON_Q_CRIT[[as.character(n)]][col]
  list(Q = Q, Q_critical = Q_crit, candidate = cand,
       reject_normal = Q > Q_crit, alpha = alpha)
}

iqr_rule_scratch <- function(x, k = 1.5) {
  q1 <- quantile(x, 0.25, names = FALSE); q3 <- quantile(x, 0.75, names = FALSE)
  iqr <- q3 - q1; lo <- q1 - k * iqr; hi <- q3 + k * iqr
  list(Q1 = q1, Q3 = q3, IQR = iqr, lower_fence = lo, upper_fence = hi,
       flagged = which(x < lo | x > hi))
}

# Library: outliers::grubbs.test, outliers::dixon.test, EnvStats::rosnerTest (generalized ESD)
library_demo <- function(x) {
  if (requireNamespace("outliers", quietly = TRUE)) {
    cat("outliers::grubbs.test:\n"); print(outliers::grubbs.test(x))
  } else cat("(install 'outliers' for grubbs.test / dixon.test)\n")
  if (requireNamespace("EnvStats", quietly = TRUE)) {
    cat("\nEnvStats::rosnerTest (k = 3):\n")
    print(EnvStats::rosnerTest(x, k = 3))
  }
}

if (sys.nframe() == 0) {
  x <- c(10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 80)
  cat("=== Grubbs (two-sided) ===\n");       print(grubbs_scratch(x))
  cat("\n=== Generalized ESD (r = 3) ===\n"); print(generalized_esd(x, r = 3))
  cat("\n=== Dixon Q (n = 8) ===\n");         print(dixons_q_scratch(c(10,12,14,15,16,17,19,50)))
  cat("\n=== Tukey IQR rule ===\n");          print(iqr_rule_scratch(x))
  cat("\n--- library ---\n");                 library_demo(x)
}
