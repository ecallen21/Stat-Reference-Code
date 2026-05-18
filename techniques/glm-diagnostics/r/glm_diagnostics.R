# GLM diagnostics (Reference §7.40, §7.41, §7.55)
# From-scratch; library: ResourceSelection::hoslem.test, DHARMa::simulateResiduals.
# Run with:  Rscript glm_diagnostics.R

pearson_residuals_scratch <- function(y, mu, variance_fn)
  (y - mu) / sqrt(pmax(variance_fn(mu), 1e-15))

deviance_residuals_binomial_scratch <- function(y, mu) {
  mu <- pmin(pmax(mu, 1e-12), 1 - 1e-12)
  d <- 2 * (y * log(y / mu + 1e-12) + (1 - y) * log((1 - y) / (1 - mu) + 1e-12))
  sign(y - mu) * sqrt(pmax(d, 0))
}

deviance_residuals_poisson_scratch <- function(y, mu) {
  d <- 2 * (ifelse(y > 0, y * log(y / pmax(mu, 1e-12)), 0) - (y - mu))
  sign(y - mu) * sqrt(pmax(d, 0))
}

hosmer_lemeshow_scratch <- function(y, p_hat, g = 10) {
  order_idx <- order(p_hat); p_s <- p_hat[order_idx]; y_s <- y[order_idx]
  breaks <- floor(seq(0, length(y), length.out = g + 1))
  chi2 <- 0
  for (k in seq_len(g)) {
    sl <- (breaks[k] + 1):breaks[k + 1]
    obs1 <- sum(y_s[sl]); exp1 <- sum(p_s[sl])
    nk <- length(sl); obs0 <- nk - obs1; exp0 <- nk - exp1
    if (exp1 > 0 && exp0 > 0)
      chi2 <- chi2 + (obs1 - exp1)^2 / exp1 + (obs0 - exp0)^2 / exp0
  }
  df <- g - 2
  list(chi_square = chi2, df = df,
       p_value = pchisq(chi2, df, lower.tail = FALSE),
       g = g, method = "Hosmer-Lemeshow")
}

randomized_quantile_residuals <- function(y, mu, family = c("poisson", "binomial", "nbinom"),
                                          theta = NULL, seed = 0) {
  set.seed(seed); family <- match.arg(family)
  if (family == "poisson") {
    F_lo <- ppois(y - 1, mu); F_hi <- ppois(y, mu)
  } else if (family == "binomial") {
    F_lo <- ifelse(y == 0, 0, 1 - mu); F_hi <- ifelse(y == 0, 1 - mu, 1)
  } else {
    if (is.null(theta)) stop("theta required for nbinom RQR")
    F_lo <- pnbinom(y - 1, size = theta, mu = mu)
    F_hi <- pnbinom(y, size = theta, mu = mu)
  }
  u <- runif(length(y), F_lo, F_hi)
  qnorm(pmin(pmax(u, 1e-15), 1 - 1e-15))
}

if (sys.nframe() == 0) {
  set.seed(9); n <- 300
  x <- rnorm(n); eta <- 0.2 + 0.8 * x; mu <- 1 / (1 + exp(-eta))
  y <- as.integer(runif(n) < mu)
  cat("Pearson residuals:  mean =", mean(pearson_residuals_scratch(y, mu, function(m) m * (1 - m))), "\n")
  cat("Deviance residuals: mean =", mean(deviance_residuals_binomial_scratch(y, mu)), "\n\n")
  cat("=== Hosmer-Lemeshow ===\n"); print(hosmer_lemeshow_scratch(y, mu))
  mu_p <- exp(0.5 + 0.6 * x); yp <- rpois(n, mu_p)
  rqr <- randomized_quantile_residuals(yp, mu_p, family = "poisson")
  cat("\n=== RQR (Poisson) ===\n")
  cat("  mean =", mean(rqr), "  sd =", sd(rqr), "\n")
}
