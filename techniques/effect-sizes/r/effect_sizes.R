# Effect size measures (Reference §1.6 and §1.25)
# From-scratch base-R plus package calls (effectsize, effsize, rstatix).
# Run with:  Rscript effect_sizes.R
#
# Inputs used below:
#   x1, x2             : numeric vectors -- the two independent samples (group 1, group 2)
#   treatment, control : two-sample inputs for Glass's delta (control SD is the scale)
#   groups             : list of group vectors (one entry per group) for one-way ANOVA effect sizes
#   u1                 : Mann-Whitney U for group 1 (count of (x1 > x2) pairs, ties contribute 0.5)
#   value, kind        : magnitude and effect-size family for the verbal interpret() label
#                        kind in {"d", "r", "eta2", "f"}

mean_ <- function(x) sum(x) / length(x)
var_  <- function(x) { m <- mean_(x); sum((x - m)^2) / (length(x) - 1) }

cohens_d_scratch <- function(x1, x2) {
  n1 <- length(x1); n2 <- length(x2)
  sp2 <- ((n1 - 1) * var_(x1) + (n2 - 1) * var_(x2)) / (n1 + n2 - 2)
  (mean_(x1) - mean_(x2)) / sqrt(sp2)
}

hedges_g_scratch <- function(x1, x2) {
  df <- length(x1) + length(x2) - 2
  cohens_d_scratch(x1, x2) * (1 - 3 / (4 * df - 1))
}

glass_delta_scratch <- function(treatment, control)
  (mean_(treatment) - mean_(control)) / sqrt(var_(control))

cliffs_delta_scratch <- function(x1, x2) {
  m <- outer(x1, x2, FUN = function(a, b) sign(a - b))
  sum(m) / (length(x1) * length(x2))
}

rank_biserial_from_u <- function(u1, n1, n2) 2 * u1 / (n1 * n2) - 1

eta_squared_oneway <- function(groups) {
  allv <- unlist(groups); grand <- mean_(allv); N <- length(allv); k <- length(groups)
  ss_b <- sum(vapply(groups, function(g) length(g) * (mean_(g) - grand)^2, numeric(1)))
  ss_t <- sum((allv - grand)^2); ss_w <- ss_t - ss_b; ms_w <- ss_w / (N - k)
  eta2 <- ss_b / ss_t
  list(eta_squared = eta2,
       omega_squared = (ss_b - (k - 1) * ms_w) / (ss_t + ms_w),
       cohens_f = sqrt(eta2 / (1 - eta2)))
}

interpret <- function(value, kind = c("d", "r", "eta2", "f")) {
  kind <- match.arg(kind); v <- abs(value)
  cuts <- switch(kind,
    d = c(0.2, 0.5, 0.8), r = c(0.1, 0.3, 0.5),
    eta2 = c(0.01, 0.06, 0.14), f = c(0.1, 0.25, 0.4))
  c("negligible", "small", "medium", "large")[findInterval(v, cuts) + 1]
}

# Library: effectsize::cohens_d / hedges_g / glass_delta / eta_squared,
#          effsize::cohen.d (also Cliff's delta via effsize::cliff.delta), rstatix
library_versions <- function(x1, x2) {
  out <- list()
  if (requireNamespace("effsize", quietly = TRUE)) {
    out$`cohen.d (effsize)` <- effsize::cohen.d(x1, x2)$estimate
    out$`cliff.delta (effsize)` <- effsize::cliff.delta(x1, x2)$estimate
  }
  if (requireNamespace("effectsize", quietly = TRUE))
    out$`hedges_g (effectsize)` <- as.numeric(effectsize::hedges_g(x1, x2)$Hedges_g)
  if (length(out) == 0) out$note <- "install 'effsize'/'effectsize' for library versions"
  out
}

if (sys.nframe() == 0) {
  set.seed(1)
  a <- round(rnorm(30, 105, 15), 1); b <- round(rnorm(28, 100, 15), 1)
  cat(sprintf("group A: n=%d mean=%.2f   group B: n=%d mean=%.2f\n\n",
              length(a), mean(a), length(b), mean(b)))
  d <- cohens_d_scratch(a, b); g <- hedges_g_scratch(a, b); cd <- cliffs_delta_scratch(a, b)
  cat("--- from scratch (two groups) ---\n")
  cat(sprintf("Cohen's d     : %.4f  (%s)\n", d, interpret(d, "d")))
  cat(sprintf("Hedges' g     : %.4f  (%s)\n", g, interpret(g, "d")))
  cat(sprintf("Glass's delta : %.4f\n", glass_delta_scratch(a, b)))
  cat(sprintf("Cliff's delta : %.4f  (%s)\n", cd, interpret(cd, "r")))
  c3 <- round(rnorm(25, 110, 15), 1); es <- eta_squared_oneway(list(a, b, c3))
  cat("\n--- from scratch (one-way ANOVA, 3 groups) ---\n")
  cat(sprintf("eta^2         : %.4f  (%s)\n", es$eta_squared, interpret(es$eta_squared, "eta2")))
  cat(sprintf("omega^2       : %.4f\n", es$omega_squared))
  cat(sprintf("Cohen's f     : %.4f  (%s)\n", es$cohens_f, interpret(es$cohens_f, "f")))
  cat("\n--- library ---\n"); print(library_versions(a, b))
}
