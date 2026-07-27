# Parametric bootstrap (Reference §10.2)
# From-scratch base R + boot::boot(sim="parametric") as library cross-check.
# Run with:  Rscript parametric_bootstrap.R

parametric_bootstrap <- function(x, fit_fn, sample_fn, statistic,
                                  n_boot = 2000, conf = 0.95, seed = 0) {
  set.seed(seed); n <- length(x); params <- fit_fn(x); theta_hat <- statistic(x)
  theta_star <- replicate(n_boot, statistic(sample_fn(params, n)))
  alpha <- 1 - conf
  q <- quantile(theta_star, c(alpha / 2, 1 - alpha / 2))
  z <- qnorm(1 - alpha / 2); se <- sd(theta_star)
  list(theta_hat = theta_hat, fitted_params = params, bootstrap_SE = se,
       CI_percentile = c(lower = q[[1]], upper = q[[2]]),
       CI_basic = c(lower = 2 * theta_hat - q[[2]], upper = 2 * theta_hat - q[[1]]),
       CI_normal = c(lower = theta_hat - z * se, upper = theta_hat + z * se),
       n_boot = n_boot, n = n)
}

if (sys.nframe() == 0) {
  set.seed(7); x <- rgamma(80, shape = 2.5, scale = 1.3)

  fit_gamma <- function(x) {
    m <- mean(x); v <- var(x)
    shape <- m^2 / v; scale <- v / m       # method-of-moments starting values
    # refine via MLE using fitdistr if available
    if (requireNamespace("MASS", quietly = TRUE)) {
      f <- suppressWarnings(MASS::fitdistr(x, "gamma"))
      list(shape = unname(f$estimate["shape"]), scale = 1 / unname(f$estimate["rate"]))
    } else list(shape = shape, scale = scale)
  }
  sample_gamma <- function(params, n) rgamma(n, shape = params$shape, scale = params$scale)
  shape_stat <- function(z) fit_gamma(z)$shape

  cat("=== Parametric bootstrap of gamma shape (true = 2.5) ===\n")
  print(parametric_bootstrap(x, fit_gamma, sample_gamma, shape_stat, n_boot = 1000))
}
