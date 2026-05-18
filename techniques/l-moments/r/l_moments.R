# L-moments and probability-weighted moments (Reference §1.24)
# From-scratch base-R plus package calls (lmom, lmomco, Lmoments). Run:  Rscript l_moments.R
#
# Inputs used below:
#   x         : numeric vector (the sample; needs n > max_order observations)
#   max_order : highest PWM order to compute (default 3 -> returns b0..b3,
#               which gives L1..L4 and the L-CV / L-skewness / L-kurtosis ratios)

# Unbiased sample probability-weighted moments b0..b_maxorder (Landwehr et al.)
pwm_scratch <- function(x, max_order = 3) {
  s <- sort(as.numeric(x)); n <- length(s)
  stopifnot(n > max_order)
  vapply(0:max_order, function(r) {
    denom <- choose(n - 1, r)
    sum(choose(seq_len(n) - 1, r) * s) / (n * denom)
  }, numeric(1))
}

l_moments_scratch <- function(x) {
  b <- pwm_scratch(x, 3); b0 <- b[1]; b1 <- b[2]; b2 <- b[3]; b3 <- b[4]
  l1 <- b0
  l2 <- 2 * b1 - b0
  l3 <- 6 * b2 - 6 * b1 + b0
  l4 <- 20 * b3 - 30 * b2 + 12 * b1 - b0
  c(L1 = l1, L2 = l2, L3 = l3, L4 = l4,
    L_CV = l2 / l1, L_skewness = l3 / l2, L_kurtosis = l4 / l2)
}

# Library: lmom::samlmu (sample L-moments + ratios), lmomco::lmoms, Lmoments::Lmoments
library_demo <- function(x) {
  if (requireNamespace("lmom", quietly = TRUE))
    print(lmom::samlmu(x, nmom = 4, ratios = TRUE))
  else cat("(install 'lmom' for lmom::samlmu)\n")
}

if (sys.nframe() == 0) {
  set.seed(0)
  data <- sort(round(rlnorm(40, meanlog = 3.0, sdlog = 0.7), 1))
  cat("n =", length(data), " (right-skewed sample)\n\n--- probability-weighted moments ---\n")
  b <- pwm_scratch(data); for (r in seq_along(b)) cat(sprintf("b%d = %.4f\n", r - 1, b[r]))
  cat("\n--- L-moments ---\n"); print(round(l_moments_scratch(data), 4))
  cat("\n--- library (lmom::samlmu) ---\n"); library_demo(data)
}
