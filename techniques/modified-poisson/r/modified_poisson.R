# Modified Poisson regression for risk ratios (Reference §7.9, §7.53)
# From-scratch IRLS + Huber-White sandwich SEs; library: sandwich::vcovHC.
# Run with:  Rscript modified_poisson.R

modified_poisson_scratch <- function(X, y) {
  X <- as.matrix(X); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  beta <- rep(0, p); ll_prev <- -Inf
  for (it in seq_len(50)) {
    eta <- as.vector(X %*% beta); mu <- exp(pmin(pmax(eta, -500), 500))
    w <- pmax(mu, 1e-12); z <- eta + (y - mu) / w
    sw <- sqrt(w); Xw <- sw * X; zw <- sw * z
    beta_new <- as.vector(solve(crossprod(Xw), crossprod(Xw, zw)))
    ll <- sum(y * log(pmax(mu, 1e-15)) - mu)
    if (abs(ll - ll_prev) < 1e-8) break
    ll_prev <- ll; beta <- beta_new
  }
  eta <- as.vector(X %*% beta); mu <- exp(eta); e <- y - mu
  XtWX <- t(X) %*% (X * mu); A_inv <- solve(XtWX)
  naive_se <- sqrt(diag(A_inv))
  B <- t(X) %*% (X * e^2)
  var_robust <- A_inv %*% B %*% A_inv * n / (n - p)
  se_robust <- sqrt(diag(var_robust))
  list(beta = beta, naive_poisson_se = naive_se,
       robust_se = se_robust, z = beta / se_robust,
       p_values = 2 * pnorm(-abs(beta / se_robust)),
       risk_ratio = exp(beta),
       rr_ci_lower = exp(beta - qnorm(0.975) * se_robust),
       rr_ci_upper = exp(beta + qnorm(0.975) * se_robust))
}

library_demo <- function(X, y) {
  fit <- glm(y ~ X[, -1], family = poisson(link = "log"))
  cat("Naive Poisson SEs:\n"); print(summary(fit)$coefficients[, "Std. Error"])
  if (requireNamespace("sandwich", quietly = TRUE)
      && requireNamespace("lmtest", quietly = TRUE)) {
    cat("\nsandwich::vcovHC (HC1) -> robust SEs:\n")
    print(lmtest::coeftest(fit, vcov = sandwich::vcovHC(fit, type = "HC1")))
  }
}

if (sys.nframe() == 0) {
  set.seed(6); n <- 500
  x1 <- rnorm(n); x2 <- rnorm(n)
  p_true <- pmin(exp(-0.4 + 0.4 * x1 - 0.3 * x2), 1)
  y <- as.integer(runif(n) < p_true); X <- cbind(1, x1, x2)
  print(modified_poisson_scratch(X, y))
  cat("\n--- library ---\n"); library_demo(X, y)
}
