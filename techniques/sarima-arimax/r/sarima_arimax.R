# SARIMA + ARIMAX (Reference §13.6, §13.25)
# Base R via stats::arima and forecast::Arima with seasonal + xreg.
# Run with:  Rscript sarima_arimax.R

if (sys.nframe() == 0) {
  set.seed(17); n <- 240; s <- 12
  t <- 1:n
  seasonal <- 3 * sin(2 * pi * t / s); trend <- 0.02 * t
  noise <- as.numeric(arima.sim(model = list(ar = 0.6), n = n))
  y <- ts(trend + seasonal + noise, frequency = s)
  cat("=== SARIMA(1,1,1)(1,1,1,12) ===\n")
  fit <- arima(y, order = c(1, 1, 1), seasonal = list(order = c(1, 1, 1), period = s))
  print(fit)
  cat("\n=== ARIMAX with Fourier seasonal terms ===\n")
  exog <- cbind(sin = sin(2 * pi * t / s), cos = cos(2 * pi * t / s))
  fit2 <- arima(y, order = c(1, 1, 1), xreg = exog)
  print(fit2)
  if (requireNamespace("forecast", quietly = TRUE)) {
    cat("\n=== forecast::auto.arima with seasonal ===\n")
    print(forecast::auto.arima(y))
  }
}
