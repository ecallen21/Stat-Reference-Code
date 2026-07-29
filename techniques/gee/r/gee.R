# Generalized Estimating Equations (Reference §12.8, §12.24, §12.31)
# Base R via geepack::geeglm.
# Run with:  Rscript gee.R

if (sys.nframe() == 0) {
  set.seed(23); n_cl <- 50; n_per <- 6; n <- n_cl * n_per
  cluster <- rep(1:n_cl, each = n_per)
  u <- rnorm(n_cl, 0, 0.6); x <- rnorm(n)
  eta <- -0.2 + 0.5 * x + u[cluster]
  p_prob <- 1 / (1 + exp(-eta))
  y <- as.integer(runif(n) < p_prob)
  df <- data.frame(y = y, x = x, cluster = factor(cluster))
  if (requireNamespace("geepack", quietly = TRUE)) {
    cat("=== geepack::geeglm (binomial, exchangeable) ===\n")
    print(summary(geepack::geeglm(y ~ x, id = cluster, data = df,
                                    family = binomial(link = "logit"), corstr = "exchangeable")))
  }
}
