# Bayesian A/B testing (Reference §14.33)
# Base R: Beta-Binomial posteriors + Monte Carlo comparisons.
# Run with:  Rscript bayesian_ab_testing.R

if (sys.nframe() == 0) {
  y_A <- 120; n_A <- 1000; y_B <- 145; n_B <- 1000
  a_A <- 1 + y_A; b_A <- 1 + n_A - y_A
  a_B <- 1 + y_B; b_B <- 1 + n_B - y_B
  N <- 100000
  p_A <- rbeta(N, a_A, b_A); p_B <- rbeta(N, a_B, b_B)
  cat(sprintf("P(B > A)          = %.4f\n", mean(p_B > p_A)))
  cat(sprintf("mean lift (B - A) = %.4f  95%% CI (%.4f, %.4f)\n",
              mean(p_B - p_A), quantile(p_B - p_A, 0.025), quantile(p_B - p_A, 0.975)))
  cat(sprintf("expected loss if pick B = %.5f\n", mean(pmax(p_A - p_B, 0))))
  cat(sprintf("expected loss if pick A = %.5f\n", mean(pmax(p_B - p_A, 0))))
}
