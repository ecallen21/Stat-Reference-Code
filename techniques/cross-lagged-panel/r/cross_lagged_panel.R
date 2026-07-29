# Cross-Lagged Panel Model + RI-CLPM (Reference §12.10, §12.18)
# Classic CLPM via two OLS; RI-CLPM via lavaan SEM (recommended).
# Run with:  Rscript cross_lagged_panel.R

clpm_two_wave <- function(X1, Y1, X2, Y2) {
  m_x <- lm(X2 ~ X1 + Y1); m_y <- lm(Y2 ~ Y1 + X1)
  list(X_regression = summary(m_x), Y_regression = summary(m_y))
}

if (sys.nframe() == 0) {
  set.seed(41); n <- 300
  X1 <- rnorm(n); Y1 <- 0.3 * X1 + rnorm(n)
  X2 <- 0.6 * X1 + rnorm(n); Y2 <- 0.5 * Y1 + 0.4 * X1 + rnorm(n)
  cat("=== Classic 2-wave CLPM ===\n"); print(clpm_two_wave(X1, Y1, X2, Y2))
  if (requireNamespace("lavaan", quietly = TRUE)) {
    cat("\n=== RI-CLPM via lavaan (needs 3+ waves) ===\n")
    cat("Sketch: define latent RI_X, RI_Y; wt-specific residuals a_x, a_y;\n")
    cat("        cross-lagged paths from a_x_t to a_y_{t+1} etc.\n")
    cat("See lavaan tutorial for the full model syntax.\n")
  }
}
