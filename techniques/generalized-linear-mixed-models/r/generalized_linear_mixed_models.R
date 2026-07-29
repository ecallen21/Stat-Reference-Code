# Generalized Linear Mixed Models (Reference §12.3; also covers §12.23 ordinal GLMM)
# Base R via lme4::glmer (binomial / Poisson / etc.) and ordinal::clmm (cumulative-link).
# Run with:  Rscript generalized_linear_mixed_models.R

if (sys.nframe() == 0) {
  set.seed(19); n_cl <- 40; n_per <- 10; n <- n_cl * n_per
  cluster <- rep(1:n_cl, each = n_per)
  u <- rnorm(n_cl, 0, 0.7); x <- rnorm(n)
  eta <- -0.3 + 0.8 * x + u[cluster]
  p_prob <- 1 / (1 + exp(-eta))
  y <- as.integer(runif(n) < p_prob)
  df <- data.frame(y = y, x = x, cluster = factor(cluster))
  if (requireNamespace("lme4", quietly = TRUE)) {
    cat("=== lme4::glmer (binomial, random intercept) ===\n")
    print(summary(lme4::glmer(y ~ x + (1 | cluster), data = df, family = binomial())))
  }
  # Ordinal GLMM demo (§12.23)
  if (requireNamespace("ordinal", quietly = TRUE)) {
    ord_y <- factor(cut(qnorm(runif(n)) + 0.4 * x + u[cluster], breaks = c(-Inf, -0.5, 0.5, Inf)),
                     labels = c("low", "med", "high"))
    df2 <- data.frame(y = ord_y, x = x, cluster = factor(cluster))
    cat("\n=== ordinal::clmm (cumulative-link mixed, random intercept) ===\n")
    print(summary(ordinal::clmm(y ~ x + (1 | cluster), data = df2)))
  }
}
