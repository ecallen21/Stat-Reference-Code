# Hamiltonian Monte Carlo (Reference §14.8)
# Base R leapfrog HMC.  For production HMC/NUTS use rstan, cmdstanr,
# or the R interface to PyMC/Turing.
# Run with:  Rscript hamiltonian_mc.R

hmc <- function(log_target, grad_log_target, theta0, n_iter = 2000,
                eps = 0.2, L = 20, seed = 0) {
  set.seed(seed); d <- length(theta0)
  samples <- matrix(0, n_iter, d)
  theta <- theta0; accept <- 0
  U <- function(q) -log_target(q); grad_U <- function(q) -grad_log_target(q)
  for (t in 1:n_iter) {
    p <- rnorm(d); q <- theta; q_cur <- q; p_cur <- p
    p <- p - 0.5 * eps * grad_U(q)
    for (l in 1:(L - 1)) {
      q <- q + eps * p
      p <- p - eps * grad_U(q)
    }
    q <- q + eps * p
    p <- p - 0.5 * eps * grad_U(q)
    H_new <- U(q) + 0.5 * sum(p^2)
    H_cur <- U(q_cur) + 0.5 * sum(p_cur^2)
    if (log(runif(1)) < H_cur - H_new) { theta <- q; accept <- accept + 1 }
    samples[t, ] <- theta
  }
  list(samples = samples, acceptance = accept / n_iter)
}

if (sys.nframe() == 0) {
  Sigma <- matrix(c(1, 0.9, 0.9, 1), 2, 2); Sigma_inv <- solve(Sigma)
  log_target <- function(q) as.numeric(-0.5 * t(q) %*% Sigma_inv %*% q)
  grad_log   <- function(q) as.numeric(-Sigma_inv %*% q)
  r <- hmc(log_target, grad_log, theta0 = c(0, 0), n_iter = 2000, eps = 0.2, L = 20)
  S <- r$samples[501:2000, ]
  cat(sprintf("HMC on 2-D correlated Gaussian: acceptance = %.3f\n", r$acceptance))
  cat("Empirical covariance:\n"); print(round(cov(S), 3))
  cat("True covariance:\n"); print(Sigma)
}
