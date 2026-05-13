# Equivalence testing via Two One-Sided Tests (TOST) (Reference §3.21)
# From-scratch base R plus equivalence/TOSTER package calls.  Run with:  Rscript tost.R
#
# Inputs used below:
#   x, x1, x2    : numeric vectors
#   lower, upper : equivalence bounds for the mean (or mean difference)
#   alpha        : per-test significance level (commonly 0.05 -> 90% CI rule)
#   equal_var    : Welch (FALSE, default) or Student (TRUE) two-sample test

tost_one_sample <- function(x, lower, upper, alpha = 0.05) {
  n <- length(x); m <- mean(x); s <- sd(x); se <- s / sqrt(n); df <- n - 1
  t_lo <- (m - lower) / se;  t_up <- (m - upper) / se
  p_lo <- pt(t_lo, df, lower.tail = FALSE)
  p_up <- pt(t_up, df, lower.tail = TRUE)
  p_tost <- max(p_lo, p_up)
  tc <- qt(1 - alpha, df)
  list(mean = m, se = se, df = df,
       t_lower = t_lo, p_lower = p_lo,
       t_upper = t_up, p_upper = p_up,
       p_tost = p_tost, equivalent = p_tost < alpha,
       ci_inner_low = m - tc * se, ci_inner_high = m + tc * se,
       margin = c(lower, upper))
}

tost_two_sample <- function(x1, x2, lower, upper, alpha = 0.05, equal_var = FALSE) {
  n1 <- length(x1); n2 <- length(x2)
  m1 <- mean(x1); m2 <- mean(x2); v1 <- var(x1); v2 <- var(x2)
  diff <- m1 - m2
  if (equal_var) {
    sp2 <- ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    se <- sqrt(sp2 * (1/n1 + 1/n2)); df <- n1 + n2 - 2
  } else {
    se <- sqrt(v1/n1 + v2/n2)
    df <- (v1/n1 + v2/n2)^2 / ((v1/n1)^2 / (n1 - 1) + (v2/n2)^2 / (n2 - 1))
  }
  t_lo <- (diff - lower) / se; t_up <- (diff - upper) / se
  p_lo <- pt(t_lo, df, lower.tail = FALSE)
  p_up <- pt(t_up, df, lower.tail = TRUE)
  p_tost <- max(p_lo, p_up); tc <- qt(1 - alpha, df)
  list(mean_diff = diff, se = se, df = df,
       t_lower = t_lo, p_lower = p_lo,
       t_upper = t_up, p_upper = p_up,
       p_tost = p_tost, equivalent = p_tost < alpha,
       ci_inner_low = diff - tc * se, ci_inner_high = diff + tc * se,
       margin = c(lower, upper),
       method = if (equal_var) "Student" else "Welch")
}

tost_paired <- function(x1, x2, lower, upper, alpha = 0.05) {
  stopifnot(length(x1) == length(x2))
  tost_one_sample(x1 - x2, lower, upper, alpha)
}

# Library: TOSTER (TOSTER::TOSTtwo, TOSTER::TOSTpaired, ...),
#          equivalence (equivalence::tost)
library_demo <- function(x1, x2, lower, upper) {
  if (requireNamespace("equivalence", quietly = TRUE)) {
    cat("equivalence::tost:\n")
    print(equivalence::tost(x1, x2, epsilon = upper, var.equal = FALSE))
  } else cat("(install 'equivalence' or 'TOSTER' for library calls)\n")
}

if (sys.nframe() == 0) {
  set.seed(13)
  a <- rnorm(40, 50, 8); b <- rnorm(38, 50.4, 8)
  cat("=== Two-sample TOST  margin = (-2, +2)  alpha = 0.05 ===\n")
  print(tost_two_sample(a, b, -2, 2))
  pre <- rnorm(25, 100, 12); post <- pre + rnorm(25, 0.5, 4)
  cat("\n=== Paired TOST  margin = (-3, +3) ===\n")
  print(tost_paired(post, pre, -3, 3))
  cat("\n--- library ---\n"); library_demo(a, b, -2, 2)
}
