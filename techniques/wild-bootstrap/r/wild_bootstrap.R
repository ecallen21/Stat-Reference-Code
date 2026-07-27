# Wild bootstrap (Reference §10.5)
# From-scratch base R + boot::boot for a cross-check.
# Run with:  Rscript wild_bootstrap.R

wild_weights <- function(scheme, n) {
  if (scheme == "rademacher") return(sample(c(-1, 1), n, replace = TRUE))
  if (scheme == "mammen") {
    phi <- (1 + sqrt(5)) / 2; p <- phi / sqrt(5)
    a <- -(sqrt(5) - 1) / 2; b <- (sqrt(5) + 1) / 2
    return(ifelse(runif(n) < p, a, b))
  }
  if (scheme == "normal") return(rnorm(n))
  stop("scheme must be rademacher / mammen / normal")
}

wild_bootstrap_regression <- function(X, y, coef_index = NULL,
                                       weights = "mammen", n_boot = 2000,
                                       conf = 0.95, seed = 0) {
  set.seed(seed); X <- as.matrix(X); n <- nrow(X); p <- ncol(X)
  fit <- lm.fit(X, y); beta_hat <- fit$coefficients; yhat <- fit$fitted.values
  resid <- y - yhat
  beta_star <- matrix(0, n_boot, p)
  for (b in seq_len(n_boot)) {
    w <- wild_weights(weights, n)
    y_star <- yhat + resid * w
    beta_star[b, ] <- lm.fit(X, y_star)$coefficients
  }
  alpha <- 1 - conf; se <- apply(beta_star, 2, sd)
  out <- list(beta_hat = beta_hat, bootstrap_SE = se,
              weights = weights, n_boot = n_boot, n = n, p = p)
  if (!is.null(coef_index)) {
    q <- quantile(beta_star[, coef_index], c(alpha / 2, 1 - alpha / 2))
    out$coef_index <- coef_index
    out$CI_percentile <- c(lower = q[[1]], upper = q[[2]])
  }
  out
}

if (sys.nframe() == 0) {
  set.seed(29); n <- 200
  x1 <- rnorm(n); x2 <- rnorm(n)
  X <- cbind(1, x1, x2)
  y <- 1 + 0.7 * x1 - 0.4 * x2 + rnorm(n, 0, 0.5 + abs(x1))
  cat("=== Wild bootstrap (Mammen) for x1 slope ===\n")
  print(wild_bootstrap_regression(X, y, coef_index = 2, weights = "mammen"))

  cat("\n=== Comparison: naive lm() SE (homoscedastic assumption) ===\n")
  fit <- lm(y ~ x1 + x2); print(summary(fit)$coefficients)

  if (requireNamespace("sandwich", quietly = TRUE) &&
      requireNamespace("lmtest", quietly = TRUE)) {
    cat("\n=== HC1 robust SE (also handles heteroscedasticity) ===\n")
    print(lmtest::coeftest(fit, vcov = sandwich::vcovHC(fit, type = "HC1")))
  }
}
