# Metropolis-Hastings (Reference §14.6)
# Base R: random-walk MH with adaptive proposal.
# Run with:  Rscript mcmc_metropolis_hastings.R

mh <- function(log_target, theta0, n_iter = 5000, prop_sd = 0.5, seed = 0) {
  set.seed(seed); d <- length(theta0)
  samples <- matrix(0, n_iter, d)
  theta <- theta0; log_p <- log_target(theta); accept <- 0
  Sigma <- diag(d) * prop_sd^2
  for (t in 1:n_iter) {
    prop <- as.numeric(MASS::mvrnorm(1, theta, Sigma))
    log_p_prop <- log_target(prop)
    if (log(runif(1)) < log_p_prop - log_p) {
      theta <- prop; log_p <- log_p_prop; accept <- accept + 1
    }
    samples[t, ] <- theta
    if (t >= 200 && t %% 50 == 0) {
      w <- max(1, t - 500):t
      emp_cov <- cov(samples[w, , drop = FALSE])
      Sigma <- (2.38^2 / d) * (emp_cov + diag(1e-6, d))
    }
  }
  list(samples = samples, acceptance = accept / n_iter)
}

if (sys.nframe() == 0) {
  log_target <- function(theta) -0.5 * ((theta[1] - 3) / 1.5)^2
  r <- mh(log_target, theta0 = 0, n_iter = 5000, prop_sd = 0.5)
  x <- r$samples[501:5000, 1]
  cat(sprintf("MH on N(3, 1.5^2): acceptance = %.3f, mean = %.3f, sd = %.3f\n",
              r$acceptance, mean(x), sd(x)))

  if (requireNamespace("coda", quietly = TRUE)) {
    cat("=== coda diagnostics ===\n")
    print(coda::effectiveSize(coda::mcmc(x)))
  }
}
