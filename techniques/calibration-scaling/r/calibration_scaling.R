# Probability calibration (Reference §26.15)
# R via CalibrationCurves or rms::val.prob.
# Run with:  Rscript calibration_scaling.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 2000
  y <- rbinom(n, 1, 0.4)
  raw <- 3 * (y - 0.5) + rnorm(n)
  p_over <- plogis(3 * raw)
  cat(sprintf("Brier (raw) = %.4f\n", mean((p_over - y)^2)))
  # Platt scaling via logistic regression on the score
  fit <- glm(y ~ raw, family = binomial)
  p_platt <- predict(fit, type = "response")
  cat(sprintf("Brier (Platt) = %.4f\n", mean((p_platt - y)^2)))
  # Isotonic via stats::isoreg
  ord <- order(raw); iso <- isoreg(raw[ord], y[ord])
  cat(sprintf("Brier (Isotonic) = %.4f\n",
              mean((approx(sort(raw), iso$yf, xout = raw)$y - y)^2)))
}
