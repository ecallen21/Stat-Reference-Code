# Overdispersion tests for count GLMs (Reference §7.35, §7.42, §7.54)
# From-scratch + library: AER::dispersiontest, performance::check_overdispersion.
# Run with:  Rscript overdispersion_tests.R

poisson_fit_irls <- function(X, y, max_iter = 50, tol = 1e-8) {
  X <- as.matrix(X); y <- as.numeric(y)
  beta <- as.vector(solve(crossprod(X), crossprod(X, log(pmax(y, 0.5)))))
  for (it in seq_len(max_iter)) {
    eta <- as.vector(X %*% beta); mu <- exp(pmin(pmax(eta, -500), 500))
    w <- pmax(mu, 1e-12); z <- eta + (y - mu) / w
    sw <- sqrt(w); Xw <- sw * X; zw <- sw * z
    beta_new <- as.vector(solve(crossprod(Xw), crossprod(Xw, zw)))
    if (max(abs(beta_new - beta)) < tol) { beta <- beta_new; break }
    beta <- beta_new
  }
  list(beta = beta, mu = exp(as.vector(X %*% beta)))
}

pearson_dispersion_scratch <- function(X, y) {
  fit <- poisson_fit_irls(X, y); mu <- fit$mu
  chi2 <- sum((y - mu)^2 / pmax(mu, 1e-12)); n <- nrow(X); p <- ncol(X)
  list(pearson_chi_square = chi2, df = n - p, phi_hat = chi2 / (n - p))
}

score_test_scratch <- function(X, y, form = "NB2") {
  fit <- poisson_fit_irls(X, y); mu <- fit$mu
  g <- if (form == "NB2") mu else if (form == "NB1") rep(1, length(mu)) else stop()
  aux <- ((y - mu)^2 - y) / pmax(mu, 1e-12) * g / pmax(mu, 1e-12)
  T <- sum(aux) / sqrt(2 * sum((g / mu)^2))
  list(T = T, form = form,
       p_value_one_sided = pnorm(T, lower.tail = FALSE),
       p_value_two_sided = 2 * pnorm(-abs(T)))
}

if (sys.nframe() == 0) {
  set.seed(11); n <- 400
  x1 <- rnorm(n); x2 <- rnorm(n); X <- cbind(1, x1, x2)
  mu <- exp(0.5 + 0.6 * x1 - 0.3 * x2)
  cat("=== Pure Poisson data ===\n")
  print(pearson_dispersion_scratch(X, rpois(n, mu)))
  print(score_test_scratch(X, rpois(n, mu)))
  cat("\n=== NB data (theta = 1.5) ===\n")
  y_nb <- rnbinom(n, size = 1.5, mu = mu)
  print(pearson_dispersion_scratch(X, y_nb))
  print(score_test_scratch(X, y_nb))
  cat("\n--- library ---\n")
  if (requireNamespace("AER", quietly = TRUE)) {
    fit <- glm(y_nb ~ x1 + x2, family = poisson)
    cat("AER::dispersiontest (NB2):\n"); print(AER::dispersiontest(fit, trafo = 1))
  }
}
