# Bayesian GLMs (Reference §14.12, §14.13)
# Base R MH sampler on logistic and Poisson GLMs.
# Production: rstanarm::stan_glm, brms::brm, arm::bayesglm.
# Run with:  Rscript bayesian_glms.R

bayesian_logistic <- function(X, y, prior_sd = NULL, n_iter = 6000, seed = 0) {
  set.seed(seed); p <- ncol(X)
  if (is.null(prior_sd)) prior_sd <- c(10, rep(2.5, p - 1))
  neg_lp <- function(b) {
    z <- as.numeric(X %*% b)
    -sum(y * z - log1p(exp(z))) + 0.5 * sum((b / prior_sd)^2)
  }
  opt <- optim(rep(0, p), neg_lp, hessian = TRUE, method = "BFGS")
  beta_hat <- opt$par
  prop_cov <- (2.38^2 / p) * solve(opt$hessian) + diag(1e-6, p)
  L <- chol(prop_cov)
  samples <- matrix(0, n_iter, p); beta <- beta_hat; lp <- -neg_lp(beta); acc <- 0
  for (t in 1:n_iter) {
    prop <- beta + as.numeric(t(L) %*% rnorm(p))
    lp_prop <- -neg_lp(prop)
    if (log(runif(1)) < lp_prop - lp) { beta <- prop; lp <- lp_prop; acc <- acc + 1 }
    samples[t, ] <- beta
  }
  burn <- n_iter %/% 5
  list(post_mean = colMeans(samples[(burn + 1):n_iter, ]),
       post_sd   = apply(samples[(burn + 1):n_iter, ], 2, sd),
       acceptance = acc / n_iter)
}

if (sys.nframe() == 0) {
  set.seed(0); n <- 300; p <- 3
  X <- cbind(1, matrix(rnorm(n * p), n, p))
  beta_true <- c(-0.5, 1.2, -0.8, 0.4)
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- as.numeric(runif(n) < prob)
  r <- bayesian_logistic(X, y, n_iter = 6000)
  cat("=== Bayesian logistic regression ===\n")
  for (i in seq_along(r$post_mean)) {
    cat(sprintf("  beta_%d: mean = %.3f, SD = %.3f, true = %.2f\n",
                i - 1, r$post_mean[i], r$post_sd[i], beta_true[i]))
  }
  cat(sprintf("  acceptance rate: %.3f\n", r$acceptance))
}
