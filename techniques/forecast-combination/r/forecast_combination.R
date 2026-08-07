# Forecast combination (Reference §13.30)
# R via forecastHybrid or manual weights.
# Run with:  Rscript forecast_combination.R

if (sys.nframe() == 0) {
  set.seed(0); T_ <- 100; K <- 5
  true_y <- cumsum(rnorm(T_))
  bias <- rnorm(K, 0, 0.3); noise_sd <- runif(K, 0.5, 2.0)
  F <- outer(true_y, rep(1, K)) + rep(bias, each = T_) +
       matrix(rnorm(T_ * K), T_, K) * rep(noise_sd, each = T_)
  y_tr <- true_y[1:70]; y_te <- true_y[71:100]
  F_tr <- F[1:70, ]; F_te <- F[71:100, ]

  cat("=== Individual test MSE ===\n")
  print(colMeans((F_te - y_te)^2))

  cat("\n=== Simple mean MSE ===\n")
  cat(sprintf("  %.4f\n", mean((rowMeans(F_te) - y_te)^2)))

  cat("\n=== Bates-Granger weights ===\n")
  var_i <- apply(F_tr - y_tr, 2, var)
  w <- (1 / var_i) / sum(1 / var_i)
  cat(sprintf("  weights: %s\n", paste(round(w, 3), collapse = " ")))
  cat(sprintf("  MSE: %.4f\n", mean((as.numeric(F_te %*% w) - y_te)^2)))

  cat("\n=== Granger-Ramanathan (OLS) ===\n")
  fit <- lm(y_tr ~ F_tr)
  pred <- as.numeric(cbind(1, F_te) %*% coef(fit))
  cat(sprintf("  MSE: %.4f\n", mean((pred - y_te)^2)))
}
