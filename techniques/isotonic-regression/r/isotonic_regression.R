# Isotonic Regression via PAVA (Reference §5.29)
# Base R stats::isoreg.
# Run with:  Rscript isotonic_regression.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 60
  x <- seq(0, 10, length.out = n)
  y <- log1p(x) + rnorm(n, 0, 0.4)
  fit <- isoreg(x, y)
  cat(sprintf("isoreg fitted mean of RSS = %.3f\n", sum((fit$yf - y)^2)))
  cat(sprintf("RSS to true log(1+x) = %.3f\n", sum((fit$yf - log1p(x))^2)))
}
