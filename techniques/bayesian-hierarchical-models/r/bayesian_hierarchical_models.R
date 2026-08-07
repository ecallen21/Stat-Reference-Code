# Bayesian hierarchical models (Reference §14.15, §14.16)
# Base R Gibbs sampler on the classic Rubin 8-schools model.
# Production: rstanarm::stan_glmer / brms::brm.
# Run with:  Rscript bayesian_hierarchical_models.R

hier_normal_gibbs <- function(y, sigma, n_iter = 8000, seed = 0) {
  set.seed(seed); J <- length(y)
  mu <- mean(y); tau2 <- var(y); theta <- y
  thetas <- matrix(0, n_iter, J); mus <- tau2s <- numeric(n_iter)
  for (t in 1:n_iter) {
    v_j <- 1 / (1 / sigma^2 + 1 / tau2)
    m_j <- v_j * (y / sigma^2 + mu / tau2)
    theta <- rnorm(J, m_j, sqrt(v_j))
    prec <- J / tau2 + 1e-4
    mu <- rnorm(1, (sum(theta) / tau2) / prec, sqrt(1 / prec))
    a <- 0.5 + J / 2; b <- 0.5 + 0.5 * sum((theta - mu)^2)
    tau2 <- 1 / rgamma(1, a, rate = b)
    thetas[t, ] <- theta; mus[t] <- mu; tau2s[t] <- tau2
  }
  burn <- n_iter %/% 5
  list(theta_post_mean = colMeans(thetas[(burn + 1):n_iter, ]),
       mu_mean = mean(mus[(burn + 1):n_iter]),
       tau_mean = mean(sqrt(tau2s[(burn + 1):n_iter])))
}

if (sys.nframe() == 0) {
  y <- c(28, 8, -3, 7, -1, 1, 18, 12)
  sigma <- c(15, 10, 16, 11, 9, 11, 10, 18)
  r <- hier_normal_gibbs(y, sigma, n_iter = 10000)
  cat(sprintf("=== 8-schools hierarchical fit (Gibbs) ===\n"))
  cat(sprintf("  overall mu = %.3f, between-school tau = %.3f\n", r$mu_mean, r$tau_mean))
  cat("  posterior theta means: ", paste(round(r$theta_post_mean, 2), collapse = " "), "\n")
}
