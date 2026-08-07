# Gibbs sampler (Reference §14.7)
# Base R: closed-form full conditionals for Normal-InverseGamma and
# for the 8-school hierarchical Normal model.
# Run with:  Rscript gibbs_sampler.R

gibbs_normal_ig <- function(y, mu0 = 0, kappa0 = 0.01, a0 = 0.5, b0 = 0.5,
                            n_iter = 5000, seed = 0) {
  set.seed(seed); n <- length(y); ybar <- mean(y)
  mus <- numeric(n_iter); sig2s <- numeric(n_iter)
  mu <- ybar; sig2 <- var(y)
  for (t in 1:n_iter) {
    kappa_n <- kappa0 + n
    mu_n <- (kappa0 * mu0 + n * ybar) / kappa_n
    mu <- rnorm(1, mu_n, sqrt(sig2 / kappa_n))
    a_n <- a0 + n / 2; b_n <- b0 + 0.5 * sum((y - mu)^2)
    sig2 <- 1 / rgamma(1, shape = a_n, rate = b_n)
    mus[t] <- mu; sig2s[t] <- sig2
  }
  burn <- n_iter %/% 5
  list(mu = mus[(burn + 1):n_iter], sig2 = sig2s[(burn + 1):n_iter])
}

if (sys.nframe() == 0) {
  set.seed(0); y <- rnorm(30, 4, sqrt(2))
  r <- gibbs_normal_ig(y)
  cat(sprintf("Normal-IG posterior mu:  mean = %.3f, 95%% CI = (%.3f, %.3f)\n",
              mean(r$mu), quantile(r$mu, 0.025), quantile(r$mu, 0.975)))
  cat(sprintf("Normal-IG posterior sig2: mean = %.3f, 95%% CI = (%.3f, %.3f)\n",
              mean(r$sig2), quantile(r$sig2, 0.025), quantile(r$sig2, 0.975)))
}
