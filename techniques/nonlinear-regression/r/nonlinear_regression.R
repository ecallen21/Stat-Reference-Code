# Nonlinear least squares (Reference §5.13)
# Base R nls() (Gauss-Newton).  Robust alternative: nlrob() from robustbase.
# Run with:  Rscript nonlinear_regression.R

if (sys.nframe() == 0) {
  set.seed(0)
  x <- seq(0.1, 20, length.out = 40)
  y <- 5 * x / (2 + x) + rnorm(length(x), 0, 0.2)
  cat("=== Michaelis-Menten via nls() ===\n")
  fit <- nls(y ~ Vmax * x / (Km + x), start = list(Vmax = 1, Km = 1))
  print(summary(fit))
}
