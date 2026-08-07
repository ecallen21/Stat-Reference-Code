# Laplace approximation of a posterior (Reference §14.29)
# Base R optim() for MAP + numDeriv::hessian for the Hessian.
# For production nested Laplace for latent Gaussian models: INLA package.
# Run with:  Rscript laplace_approximation.R

if (sys.nframe() == 0) {
  cat("=== Bayesian logistic Laplace approximation ===\n")
  set.seed(0); n <- 200
  x <- rnorm(n); prob <- plogis(0.5 + 1.5 * x)
  y <- as.integer(runif(n) < prob)
  log_post <- function(theta) {
    z <- theta[1] + theta[2] * x
    sum(y * z - log1p(exp(z))) - 0.5 * sum(theta^2) / 100
  }
  fit <- optim(c(0, 0), function(t) -log_post(t), method = "BFGS", hessian = TRUE)
  cat(sprintf("  MAP = (%.3f, %.3f)   (true 0.5, 1.5)\n", fit$par[1], fit$par[2]))
  se <- sqrt(diag(solve(fit$hessian)))
  cat(sprintf("  SD  = (%.3f, %.3f)\n", se[1], se[2]))
  cat("\nProduction: install.packages('INLA') for nested-Laplace on latent-Gaussian models.\n")
}
