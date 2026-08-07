# Spline regression (Reference §5.12)
# R via splines::ns (natural), splines::bs (B-spline).
# Run with:  Rscript splines_regression.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 200
  x <- sort(runif(n, -3, 3))
  f_true <- sin(1.5 * x) + 0.3 * x
  y <- f_true + rnorm(n, 0, 0.3)
  knots <- quantile(x, probs = seq(0.1, 0.9, length.out = 6))

  cat("=== splines::ns (natural cubic) ===\n")
  fit_ns <- lm(y ~ splines::ns(x, knots = knots))
  cat(sprintf("  RMSE vs truth = %.4f\n", sqrt(mean((fitted(fit_ns) - f_true)^2))))

  cat("\n=== splines::bs (cubic B-spline) ===\n")
  fit_bs <- lm(y ~ splines::bs(x, knots = knots, degree = 3))
  cat(sprintf("  RMSE vs truth = %.4f\n", sqrt(mean((fitted(fit_bs) - f_true)^2))))
}
