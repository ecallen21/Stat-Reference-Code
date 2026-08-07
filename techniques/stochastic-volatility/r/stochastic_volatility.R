# Stochastic Volatility (Reference §13.40)
# R via stochvol::svsample (Kim-Shephard-Chib MCMC).
# Run with:  Rscript stochastic_volatility.R

if (sys.nframe() == 0) {
  set.seed(0); T_ <- 500
  mu_true <- -8; phi_true <- 0.97; sigma_eta <- 0.15
  h <- numeric(T_); h[1] <- mu_true
  for (t in 2:T_) h[t] <- mu_true + phi_true * (h[t - 1] - mu_true) + sigma_eta * rnorm(1)
  y <- exp(h / 2) * rnorm(T_)

  if (requireNamespace("stochvol", quietly = TRUE)) {
    cat("=== stochvol::svsample (Kim-Shephard-Chib MCMC) ===\n")
    fit <- stochvol::svsample(y, draws = 3000, burnin = 1000, quiet = TRUE)
    print(summary(fit))
  } else {
    cat("stochvol not installed; skip.\n")
  }
}
