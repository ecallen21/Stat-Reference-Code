# GARCH (Reference §13.11, §13.33)
# R via rugarch (authoritative for univariate and multivariate GARCH).
# Run with:  Rscript garch.R

if (sys.nframe() == 0) {
  set.seed(37); n <- 1000
  omega_t <- 0.05; alpha_t <- 0.1; beta_t <- 0.85
  r <- numeric(n); sigma2 <- numeric(n)
  sigma2[1] <- omega_t / (1 - alpha_t - beta_t); r[1] <- sqrt(sigma2[1]) * rnorm(1)
  for (t in 2:n) {
    sigma2[t] <- omega_t + alpha_t * r[t - 1]^2 + beta_t * sigma2[t - 1]
    r[t] <- sqrt(sigma2[t]) * rnorm(1)
  }
  if (requireNamespace("rugarch", quietly = TRUE)) {
    spec <- rugarch::ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                                  mean.model = list(armaOrder = c(0, 0)))
    cat("=== rugarch::ugarchfit ===\n"); print(rugarch::ugarchfit(spec, r))
  }
}
