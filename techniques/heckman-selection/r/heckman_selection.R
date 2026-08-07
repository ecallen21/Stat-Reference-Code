# Heckman selection model (Reference §5.21)
# R via sampleSelection::selection or sampleSelection::heckit.
# Run with:  Rscript heckman_selection.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 800; rho <- 0.7
  w <- rnorm(n); z_x <- rnorm(n)
  u <- rnorm(n); eps <- rho * u + sqrt(1 - rho^2) * rnorm(n)
  z_star <- 0.5 + 1.0 * w + u
  d <- as.integer(z_star > 0)
  y_star <- 2 + 1.5 * z_x + eps
  y <- ifelse(d == 1, y_star, NA)
  if (requireNamespace("sampleSelection", quietly = TRUE)) {
    cat("=== sampleSelection::heckit (two-step) ===\n")
    fit <- sampleSelection::heckit(d ~ w, y ~ z_x, method = "2step")
    print(summary(fit))
  }
}
