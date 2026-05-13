# Wald, Likelihood-Ratio, and Score (Rao) tests (Reference §3.18, §3.30, §3.31, §3.33)
# Run with:  Rscript wald_lrt_score.R
#
# Inputs used below:
#   theta_hat, theta_0          : MLE and null value
#   information_at_hat / _null  : Fisher information at the MLE / at the null
#   score_at_null               : score function evaluated at the null
#   loglik_hat / loglik_null    : maximized / constrained log-likelihoods
# For the binomial worked example:
#   x : observed number of successes
#   n : number of trials
#   p0: null hypothesis value of the success probability

wald_test <- function(theta_hat, theta_0, information_at_hat) {
  W <- (theta_hat - theta_0)^2 * information_at_hat
  list(statistic = W, df = 1, p_value = pchisq(W, 1, lower.tail = FALSE),
       se = sqrt(1 / information_at_hat), method = "Wald")
}

likelihood_ratio_test <- function(loglik_hat, loglik_null, df = 1) {
  G2 <- 2 * (loglik_hat - loglik_null)
  list(statistic = G2, df = df,
       p_value = pchisq(G2, df, lower.tail = FALSE),
       method = "Likelihood Ratio")
}

score_test <- function(score_at_null, information_at_null) {
  S <- score_at_null^2 / information_at_null
  list(statistic = S, df = 1, p_value = pchisq(S, 1, lower.tail = FALSE),
       method = "Score (Rao)")
}

# Worked example: binomial(p), H0: p = p0
binomial_three_tests <- function(x, n, p0) {
  p_hat <- x / n
  # 0 log 0 = 0 (boundary safe)
  xlogy <- function(a, b) if (a == 0) 0 else a * log(b)
  ll <- function(p) xlogy(x, p) + xlogy(n - x, 1 - p)
  ll_hat <- ll(p_hat); ll_null <- ll(p0)
  I_hat <- if (p_hat > 0 && p_hat < 1) n / (p_hat * (1 - p_hat)) else Inf
  I_null <- n / (p0 * (1 - p0))
  U_null <- x / p0 - (n - x) / (1 - p0)
  list(p_hat = p_hat,
       wald  = wald_test(p_hat, p0, I_hat),
       lrt   = likelihood_ratio_test(ll_hat, ll_null),
       score = score_test(U_null, I_null))
}

# Library: stats::lrtest is in 'lmtest'; for GLMs, anova(fit_full, fit_null, test = "LRT")
library_demo <- function(x, n, p0) {
  cat("prop.test (score-flavored chi^2):\n")
  print(prop.test(x, n, p = p0, correct = FALSE))
}

if (sys.nframe() == 0) {
  res <- binomial_three_tests(60, 100, 0.5)
  cat(sprintf("x = 60, n = 100, p0 = 0.5  (p_hat = %s)\n\n", res$p_hat))
  for (nm in c("wald", "lrt", "score")) {
    r <- res[[nm]]
    cat(sprintf("  %-18s: statistic = %.4f  p = %.4f\n",
                r$method, r$statistic, r$p_value))
  }
  cat("\nBoundary case  x = 0, n = 20, p0 = 0.1 :\n")
  res <- binomial_three_tests(0, 20, 0.1)
  for (nm in c("wald", "lrt", "score")) {
    r <- res[[nm]]
    cat(sprintf("  %-18s: statistic = %.4f  p = %.4f\n",
                r$method, r$statistic, r$p_value))
  }
  cat("(Wald is unreliable near the boundary -- LRT and score behave better.)\n")
  cat("\n--- library ---\n"); library_demo(60, 100, 0.5)
}
