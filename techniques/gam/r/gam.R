# Generalized Additive Models (Reference §5.14)
# R via mgcv::gam (Wood's REML/GCV smoothing).
# Run with:  Rscript gam.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 300
  x1 <- runif(n, -3, 3); x2 <- runif(n, 0, 10)
  y <- sin(1.5 * x1) + 0.3 * (x2 - 5)^2 - 3 + rnorm(n, 0, 0.5)
  df <- data.frame(y = y, x1 = x1, x2 = x2)
  if (requireNamespace("mgcv", quietly = TRUE)) {
    cat("=== mgcv::gam(y ~ s(x1) + s(x2)) ===\n")
    fit <- mgcv::gam(y ~ s(x1) + s(x2), data = df, method = "REML")
    print(summary(fit))
  }
}
