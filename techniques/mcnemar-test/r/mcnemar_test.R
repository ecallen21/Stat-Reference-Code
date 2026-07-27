# McNemar's test for paired binary outcomes (Reference §8.2, §8.18)
# From-scratch base R plus stats::mcnemar.test (idiomatic call).
# Run with:  Rscript mcnemar_test.R
#
# Inputs:
#   before, after : integer 0/1 vectors of equal length (one pair per subject)

build_paired_table <- function(before, after) {
  stopifnot(length(before) == length(after))
  a <- sum(before == 1 & after == 1)
  b <- sum(before == 1 & after == 0)
  c <- sum(before == 0 & after == 1)
  d <- sum(before == 0 & after == 0)
  matrix(c(a, c, b, d), nrow = 2, byrow = FALSE,
         dimnames = list(before = c("+", "-"), after = c("+", "-")))
}

mcnemar_scratch <- function(tbl, continuity = FALSE) {
  b <- tbl[1, 2]; c <- tbl[2, 1]
  n_disc <- b + c
  if (n_disc == 0) return(list(chi_square = 0, df = 1, p_value = 1,
                                b = b, c = c, n_discordant = 0,
                                note = "all pairs concordant; undefined"))
  diff <- abs(b - c) - if (continuity) 1 else 0
  diff <- max(0, diff)
  x2 <- diff^2 / n_disc
  list(b = b, c = c, n_discordant = n_disc,
       chi_square = x2, df = 1, p_value = pchisq(x2, 1, lower.tail = FALSE),
       continuity = continuity)
}

mcnemar_exact_scratch <- function(tbl, mid_p = FALSE) {
  b <- tbl[1, 2]; c <- tbl[2, 1]
  n <- b + c
  if (n == 0) return(list(b = b, c = c, n_discordant = 0, p_value = 1, mid_p = mid_p))
  k <- min(b, c)
  p_two <- min(1, 2 * pbinom(k, n, 0.5))
  if (mid_p) p_two <- min(1, max(0, p_two - dbinom(k, n, 0.5)))
  list(b = b, c = c, n_discordant = n, p_value = p_two, mid_p = mid_p)
}

mcnemar_odds_ratio <- function(tbl) {
  b <- tbl[1, 2]; c <- tbl[2, 1]
  b_adj <- if (b == 0) 0.5 else b
  c_adj <- if (c == 0) 0.5 else c
  or_hat <- b_adj / c_adj
  se <- sqrt(1 / b_adj + 1 / c_adj)
  z <- qnorm(0.975)
  c(OR = or_hat,
    CI95_lower = exp(log(or_hat) - z * se),
    CI95_upper = exp(log(or_hat) + z * se))
}

newcombe_paired_diff_ci <- function(tbl, conf = 0.95) {
  a <- tbl[1, 1]; b <- tbl[1, 2]; c <- tbl[2, 1]; d <- tbl[2, 2]
  n <- a + b + c + d
  p1 <- (a + b) / n; p2 <- (a + c) / n
  z <- qnorm(0.5 + conf / 2)
  wilson <- function(x, nn) {
    p <- x / nn; z2 <- z^2
    denom <- 1 + z2 / nn
    center <- (p + z2 / (2 * nn)) / denom
    half <- (z / denom) * sqrt(p * (1 - p) / nn + z2 / (4 * nn^2))
    c(lower = max(0, center - half), upper = min(1, center + half))
  }
  ci1 <- wilson(a + b, n); ci2 <- wilson(a + c, n)
  denom_phi <- (a + b) * (c + d) * (a + c) * (b + d)
  phi <- if (denom_phi > 0) (a * d - b * c) / sqrt(denom_phi) else 0
  delta <- p1 - p2
  lo <- delta - sqrt((p1 - ci1[["lower"]])^2 - 2 * phi * (p1 - ci1[["lower"]]) * (ci2[["upper"]] - p2) + (ci2[["upper"]] - p2)^2)
  hi <- delta + sqrt((ci1[["upper"]] - p1)^2 - 2 * phi * (ci1[["upper"]] - p1) * (p2 - ci2[["lower"]]) + (p2 - ci2[["lower"]])^2)
  c(diff = delta, CI_lower = max(-1, lo), CI_upper = min(1, hi))
}

if (sys.nframe() == 0) {
  set.seed(7)
  before <- as.integer(runif(300) < 0.35)
  after <- integer(300)
  for (i in seq_along(before)) {
    p_pos <- if (before[i] == 1) 0.90 else 0.15
    after[i] <- as.integer(runif(1) < p_pos)
  }
  tbl <- build_paired_table(before, after)
  cat("=== Paired table ===\n"); print(tbl)
  cat("\n=== Asymptotic ===\n"); print(mcnemar_scratch(tbl))
  cat("\n=== Continuity-corrected ===\n"); print(mcnemar_scratch(tbl, continuity = TRUE))
  cat("\n=== Exact ===\n"); print(mcnemar_exact_scratch(tbl))
  cat("\n=== Mid-p ===\n"); print(mcnemar_exact_scratch(tbl, mid_p = TRUE))
  cat("\n=== OR ===\n"); print(mcnemar_odds_ratio(tbl))
  cat("\n=== Newcombe paired diff CI ===\n"); print(newcombe_paired_diff_ci(tbl))
  cat("\n--- library: stats::mcnemar.test ---\n")
  print(mcnemar.test(tbl, correct = FALSE))
  print(mcnemar.test(tbl, correct = TRUE))
}
