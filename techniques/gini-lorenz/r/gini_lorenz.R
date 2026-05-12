# Gini coefficient and Lorenz curve (Reference §1.23)
# From-scratch base-R plus package calls (ineq, DescTools). Run:  Rscript gini_lorenz.R

lorenz_curve <- function(x) {
  s <- sort(as.numeric(x)); stopifnot(all(s >= 0)); n <- length(s)
  cum <- c(0, cumsum(s))
  data.frame(p = (0:n) / n, L = cum / cum[n + 1])
}

gini_trapezoid <- function(x) {
  lc <- lorenz_curve(x); p <- lc$p; L <- lc$L
  area_under <- sum(diff(p) * (head(L, -1) + tail(L, -1)) / 2)   # trapezoid rule
  1 - 2 * area_under
}

gini_mean_difference <- function(x, bias_corrected = FALSE) {
  s <- sort(as.numeric(x)); stopifnot(all(s >= 0)); n <- length(s)
  g <- (2 * sum(seq_len(n) * s)) / (n * sum(s)) - (n + 1) / n
  if (bias_corrected) g * n / (n - 1) else g
}

# Library: ineq::Gini (corr = TRUE for bias correction), ineq::Lc + plot.Lc,
#          DescTools::Gini, reldist::gini
library_demo <- function(x) {
  if (requireNamespace("ineq", quietly = TRUE)) {
    cat("ineq::Gini            :", ineq::Gini(x), "\n")
    cat("ineq::Gini(corr=TRUE) :", ineq::Gini(x, corr = TRUE), "\n")
  } else cat("(install 'ineq' for ineq::Gini / ineq::Lc)\n")
}

if (sys.nframe() == 0) {
  incomes <- c(10, 12, 15, 18, 20, 25, 30, 40, 60, 220)
  cat("incomes:", incomes, "\n\nLorenz curve points:\n"); print(lorenz_curve(incomes))
  cat("\nGini (trapezoid)            :", round(gini_trapezoid(incomes), 4), "\n")
  cat("Gini (mean-difference form) :", round(gini_mean_difference(incomes), 4), "\n")
  cat("Gini (bias-corrected)       :", round(gini_mean_difference(incomes, TRUE), 4), "\n")
  cat("\nSanity:\n  equal [5,5,5,5]            -> Gini =", round(gini_mean_difference(c(5,5,5,5)), 4), "\n")
  cat("  maximally unequal [0,0,0,100]-> Gini =", round(gini_mean_difference(c(0,0,0,100)), 4), "\n")
  cat("\n--- library ---\n"); library_demo(incomes)
}
