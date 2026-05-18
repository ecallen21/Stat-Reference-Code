# Gamma regression (Reference §7.25)
# From-scratch IRLS plus stats::glm(family = Gamma(link = "log")).
# Run with:  Rscript gamma_regression.R

gamma_fit_scratch <- function(X, y, max_iter = 50, tol = 1e-8) {
  stopifnot(all(y > 0))
  X <- as.matrix(X); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  beta <- as.vector(solve(crossprod(X), crossprod(X, log(y))))
  for (it in seq_len(max_iter)) {
    eta <- as.vector(X %*% beta); mu <- exp(pmin(pmax(eta, -500), 500))
    z <- eta + (y - mu) / mu
    beta_new <- as.vector(solve(crossprod(X), crossprod(X, z)))
    if (max(abs(beta_new - beta)) < tol) { beta <- beta_new; break }
    beta <- beta_new
  }
  eta <- as.vector(X %*% beta); mu <- exp(eta)
  pearson_chi2 <- sum(((y - mu) / mu)^2); phi <- pearson_chi2 / (n - p)
  var_beta <- phi * solve(crossprod(X)); se <- sqrt(diag(var_beta))
  z_stats <- beta / se
  list(beta = beta, se = se, z = z_stats,
       p_values = 2 * pnorm(-abs(z_stats)),
       exp_beta = exp(beta),
       dispersion_phi = phi,
       deviance = 2 * sum(-log(y / mu) + (y - mu) / mu),
       iterations = it)
}

if (sys.nframe() == 0) {
  set.seed(8); n <- 300
  x1 <- rnorm(n); x2 <- rnorm(n)
  mu <- exp(2 + 0.5 * x1 - 0.3 * x2); shape <- 4
  y <- rgamma(n, shape = shape, scale = mu / shape)
  X <- cbind(1, x1, x2)
  print(gamma_fit_scratch(X, y))
  cat("\n--- library ---\n")
  print(summary(glm(y ~ x1 + x2, family = Gamma(link = "log"))))
}
