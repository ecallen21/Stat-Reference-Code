# Hierarchical forecasting (Reference §13.37)
# R via hts::hts + hts::forecast (bottom-up / top-down / MinT).
# Run with:  Rscript hierarchical_forecasting.R

if (sys.nframe() == 0) {
  # Manual demo of the summing matrix + bottom-up + MinT
  hierarchy_S <- rbind(
    c(1, 1, 1, 1, 1),  # total
    c(1, 1, 0, 0, 0),  # region A
    c(0, 0, 1, 1, 1),  # region B
    diag(5)             # bottom
  )
  y_true_bot <- c(100, 120, 80, 90, 110)
  truth <- as.numeric(hierarchy_S %*% y_true_bot)
  set.seed(0)
  y_hat <- truth + rnorm(nrow(hierarchy_S), 0, 10)

  # Bottom-up
  y_bu <- as.numeric(hierarchy_S %*% y_hat[(nrow(hierarchy_S) - 4):nrow(hierarchy_S)])

  # MinT (diagonal W)
  W <- diag(c(100, 50, 50, 30, 30, 30, 30, 30))
  Winv <- solve(W)
  G <- solve(t(hierarchy_S) %*% Winv %*% hierarchy_S) %*% t(hierarchy_S) %*% Winv
  y_mint <- as.numeric(hierarchy_S %*% G %*% y_hat)

  cat("Incoherent base MSE:", mean((y_hat - truth)^2), "\n")
  cat("Bottom-up MSE      :", mean((y_bu - truth)^2), "\n")
  cat("MinT (diag) MSE    :", mean((y_mint - truth)^2), "\n")

  if (requireNamespace("hts", quietly = TRUE)) {
    cat("\nSee hts::hts and hts::forecast for the full production interface.\n")
  }
}
