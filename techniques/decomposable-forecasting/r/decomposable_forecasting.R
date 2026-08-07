# Decomposable forecasting (Reference §13.21)
# R via prophet::prophet or manual piecewise-linear + Fourier + holidays.
# Run with:  Rscript decomposable_forecasting.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 365
  t <- 0:(n - 1)
  trend_true <- 0.02 * t + pmax(t - 200, 0) * (-0.03)
  seasonal_true <- 3 * sin(2 * pi * t / 7) + 1 * cos(2 * pi * t / 7)
  holiday <- rep(0, n); holiday[c(50, 200, 300) + 1] <- 4
  y <- trend_true + seasonal_true + holiday + rnorm(n)

  if (requireNamespace("prophet", quietly = TRUE)) {
    cat("=== prophet ===\n")
    df <- data.frame(ds = seq.Date(as.Date("2024-01-01"), by = 1, length.out = n),
                     y = y)
    m <- prophet::prophet(df, daily.seasonality = FALSE,
                          yearly.seasonality = FALSE,
                          weekly.seasonality = TRUE)
    p <- predict(m, prophet::make_future_dataframe(m, periods = 30))
    cat(sprintf("  in-sample MAE: %.3f\n", mean(abs(y - p$yhat[1:n]))))
  } else {
    cat("prophet not installed; skip.\n")
  }
}
