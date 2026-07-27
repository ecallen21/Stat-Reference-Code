# Penalized Cox regression: L1 / L2 / elastic-net (Reference §11.21)
# Base R via glmnet (family = "cox"), the authoritative implementation.
# Run with:  Rscript penalized_cox.R

if (sys.nframe() == 0) {
  set.seed(41); n <- 200; p_dim <- 10
  X <- matrix(rnorm(n * p_dim), n, p_dim)
  beta_true <- c(0.7, -0.5, 0.4, rep(0, 7))
  T_event <- -log(runif(n)) / (0.1 * exp(X %*% beta_true))
  C <- runif(n, 0, 15)
  times <- pmin(T_event, C); events <- as.integer(T_event <= C)

  if (requireNamespace("glmnet", quietly = TRUE) &&
      requireNamespace("survival", quietly = TRUE)) {
    y <- survival::Surv(times, events)

    cat("=== Lasso Cox (alpha=1, cross-validated) ===\n")
    cv <- glmnet::cv.glmnet(X, y, family = "cox", alpha = 1, nfolds = 5)
    cat("  lambda.min =", cv$lambda.min, "\n")
    print(coef(cv, s = "lambda.min"))

    cat("\n=== Ridge Cox (alpha=0) at lambda = 0.05 ===\n")
    fit_r <- glmnet::glmnet(X, y, family = "cox", alpha = 0, lambda = 0.05)
    print(coef(fit_r))

    cat("\n=== Regularization path (lasso) ===\n")
    fit_path <- glmnet::glmnet(X, y, family = "cox", alpha = 1)
    print(fit_path)
  }
}
