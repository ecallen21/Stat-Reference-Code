# Posterior predictive checks (Reference §14.19)
# Base R implementation; bayesplot::ppc_dens_overlay and bayesplot::ppc_stat
# are the standard production tools.
# Run with:  Rscript posterior_predictive_checks.R

ppc_summary <- function(y, y_rep, stats = c("mean", "sd", "min", "max")) {
  fns <- list(mean = mean, sd = sd, min = min, max = max,
              median = median, kurtosis = function(x) mean((x - mean(x))^4) / var(x)^2 - 3)
  rows <- data.frame()
  for (s in stats) {
    fn <- fns[[s]]
    T_obs <- fn(y)
    T_rep <- apply(y_rep, 1, fn)
    p_B <- mean(T_rep >= T_obs)
    rows <- rbind(rows, data.frame(statistic = s,
                                    T_obs = T_obs,
                                    T_rep_mean = mean(T_rep),
                                    p_B = p_B,
                                    flag = abs(p_B - 0.5) > 0.45))
  }
  rows
}

if (sys.nframe() == 0) {
  set.seed(0); n <- 200
  y <- rcauchy(n); y <- y[abs(y) < 30]; n <- length(y)
  ybar <- mean(y); s2 <- var(y)
  S <- 1500
  sig2 <- 1 / rgamma(S, n / 2, rate = 0.5 * (n - 1) * s2)
  mu <- rnorm(S, ybar, sqrt(sig2 / n))
  y_rep <- t(sapply(seq_len(S), function(s) rnorm(n, mu[s], sqrt(sig2[s]))))
  cat("=== PPC on mis-specified Normal model over Cauchy data ===\n")
  print(ppc_summary(y, y_rep, c("mean", "sd", "min", "max", "kurtosis")))
}
