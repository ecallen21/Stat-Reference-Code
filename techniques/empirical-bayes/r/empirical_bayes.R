# Empirical Bayes (Reference §14.17, §14.18)
# Base R: EB Beta-Binomial + James-Stein shrinkage estimator.
# Run with:  Rscript empirical_bayes.R

eb_beta_binomial <- function(successes, trials) {
  p <- successes / trials; J <- length(p)
  p_bar <- mean(p); v <- var(p)
  denom <- max(v - p_bar * (1 - p_bar) / mean(trials), 1e-8)
  ab_sum <- p_bar * (1 - p_bar) / denom - 1
  alpha <- max(ab_sum * p_bar, 0.5)
  beta  <- max(ab_sum * (1 - p_bar), 0.5)
  eb <- (alpha + successes) / (alpha + beta + trials)
  list(alpha = alpha, beta = beta, hyperprior_mean = alpha / (alpha + beta),
       raw = p, eb = eb)
}

james_stein <- function(y, sigma = 1) {
  J <- length(y); ybar <- mean(y)
  shrinkage <- 1 - (J - 3) * sigma^2 / sum((y - ybar)^2)
  list(js = ybar + shrinkage * (y - ybar), shrinkage = shrinkage)
}

if (sys.nframe() == 0) {
  set.seed(0); J <- 20
  true_theta <- rbeta(J, 30, 70)
  n_trials <- sample(20:200, J, replace = TRUE)
  y <- rbinom(J, n_trials, true_theta)
  r <- eb_beta_binomial(y, n_trials)
  cat(sprintf("EB Beta-Binomial: alpha = %.2f, beta = %.2f\n", r$alpha, r$beta))
  cat(sprintf("RMSE(raw): %.4f    RMSE(EB): %.4f\n",
              sqrt(mean((r$raw - true_theta)^2)),
              sqrt(mean((r$eb  - true_theta)^2))))

  cat("\n=== James-Stein on J = 10 ===\n")
  true <- c(-2, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2, 3)
  y <- true + rnorm(length(true))
  r <- james_stein(y, sigma = 1)
  cat(sprintf("MSE(raw): %.4f    MSE(JS): %.4f    shrinkage = %.4f\n",
              mean((y - true)^2), mean((r$js - true)^2), r$shrinkage))
}
