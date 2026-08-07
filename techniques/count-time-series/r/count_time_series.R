# Count time series (Reference §13.32)
# R via tscount::tsglm for INGARCH.
# Run with:  Rscript count_time_series.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 1000
  omega <- 1; alpha <- 0.4; beta <- 0.3
  y <- integer(n); mu <- numeric(n); mu[1] <- omega / (1 - alpha - beta)
  y[1] <- rpois(1, mu[1])
  for (t in 2:n) {
    mu[t] <- omega + alpha * y[t - 1] + beta * mu[t - 1]
    y[t] <- rpois(1, mu[t])
  }
  if (requireNamespace("tscount", quietly = TRUE)) {
    cat("=== tscount::tsglm INGARCH(1,1) ===\n")
    fit <- tscount::tsglm(y, model = list(past_obs = 1, past_mean = 1),
                          distr = "poisson")
    print(summary(fit))
  }
}
