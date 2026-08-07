# Conjugate priors (Reference §14.1, §14.2, §14.3)
# Base R: analytic Beta / Gamma / Normal posterior updates.
# Run with:  Rscript conjugate_priors.R

if (sys.nframe() == 0) {
  cat("=== Beta-Binomial: 8/12 successes with Uniform(1,1) prior ===\n")
  a <- 1 + 8; b <- 1 + 12 - 8
  cat(sprintf("  posterior Beta(%d, %d);  mean = %.4f;  95%% CI = (%.4f, %.4f)\n",
              a, b, a / (a + b), qbeta(0.025, a, b), qbeta(0.975, a, b)))

  cat("\n=== Gamma-Poisson: n = 5 counts with Gamma(1, 1) prior ===\n")
  set.seed(0); y <- rpois(5, 3.5)
  a <- 1 + sum(y); b <- 1 + length(y)
  cat(sprintf("  data: %s\n", paste(y, collapse = " ")))
  cat(sprintf("  posterior Gamma(shape = %d, rate = %d);  mean = %.4f;  95%% CI = (%.4f, %.4f)\n",
              a, b, a / b, qgamma(0.025, a, rate = b), qgamma(0.975, a, rate = b)))

  cat("\n=== Normal-Normal (known sigma^2 = 4), n = 20 ===\n")
  y <- rnorm(20, 5, 2)
  sigma2 <- 4; mu0 <- 0; tau2 <- 100
  prec_post <- 1 / tau2 + length(y) / sigma2
  tau2_post <- 1 / prec_post
  mu_post <- tau2_post * (mu0 / tau2 + length(y) * mean(y) / sigma2)
  cat(sprintf("  posterior mean = %.4f, sd = %.4f\n", mu_post, sqrt(tau2_post)))
  cat(sprintf("  95%% credible interval for mu: (%.4f, %.4f)\n",
              qnorm(0.025, mu_post, sqrt(tau2_post)),
              qnorm(0.975, mu_post, sqrt(tau2_post))))
}
