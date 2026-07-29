# Generalized ordered logit (Reference §8.35)
# R via VGAM::vglm(cumulative(parallel = FALSE)) and MASS::polr for the
# proportional-odds fit + brant::brant for the Brant test.
# Run with:  Rscript generalized_ordered_logit.R

if (sys.nframe() == 0) {
  set.seed(11); n <- 400
  x1 <- rnorm(n); x2 <- rnorm(n)
  lp1 <- -0.5 - 0.8 * x1 + 0.6 * x2
  lp2 <-  0.5 - 0.2 * x1 + 0.6 * x2
  u <- runif(n)
  y <- ifelse(u < plogis(lp1), 0, ifelse(u < plogis(lp2), 1, 2))
  df <- data.frame(y = factor(y, ordered = TRUE), x1 = x1, x2 = x2)

  if (requireNamespace("VGAM", quietly = TRUE)) {
    cat("=== VGAM::vglm cumulative(parallel = FALSE) ===\n")
    fit <- VGAM::vglm(y ~ x1 + x2, family = VGAM::cumulative(parallel = FALSE, reverse = TRUE), data = df)
    print(coef(fit))
  }

  if (requireNamespace("MASS", quietly = TRUE) && requireNamespace("brant", quietly = TRUE)) {
    cat("\n=== Brant test on proportional-odds fit ===\n")
    po <- MASS::polr(y ~ x1 + x2, data = df, Hess = TRUE)
    print(brant::brant(po))
  }
}
