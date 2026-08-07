# Ridge / LASSO / Elastic Net (Reference §5.9, §5.10)
# R via glmnet (Friedman-Hastie-Tibshirani coordinate descent).
# Run with:  Rscript ridge_lasso_elasticnet.R

if (sys.nframe() == 0) {
  set.seed(0); n <- 200; p <- 20
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- numeric(p); beta_true[1:5] <- c(3, -2, 1.5, -1, 0.5)
  y <- as.numeric(X %*% beta_true + rnorm(n))
  if (requireNamespace("glmnet", quietly = TRUE)) {
    cat("=== glmnet: Ridge (alpha = 0) ===\n")
    fit_r <- glmnet::glmnet(X, y, alpha = 0, lambda = 0.5)
    print(coef(fit_r)[1:7, 1])
    cat("\n=== glmnet: LASSO (alpha = 1) ===\n")
    fit_l <- glmnet::glmnet(X, y, alpha = 1, lambda = 0.05)
    print(coef(fit_l)[1:7, 1])
    cat("\n=== glmnet: Elastic Net (alpha = 0.5) with CV lambda ===\n")
    cv <- glmnet::cv.glmnet(X, y, alpha = 0.5)
    cat(sprintf("  lambda.min = %.4f, n_nonzero = %d\n",
                cv$lambda.min, sum(coef(cv, s = "lambda.min")[-1] != 0)))
  }
}
