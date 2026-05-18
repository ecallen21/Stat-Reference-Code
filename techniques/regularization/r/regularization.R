# Regularized linear regression: ridge, lasso, elastic net (Reference §5.9, §5.17)
# From-scratch base R plus glmnet. Run with:  Rscript regularization.R
#
# Inputs used below:
#   X    : n x p matrix of predictors (we standardize internally)
#   y    : response vector
#   lam  : lambda (regularization strength)
#   alpha: elastic-net mixing in [0, 1]; 1 = lasso, 0 = ridge (scaled)

standardize <- function(X) {
  mu <- colMeans(X); s <- apply(X, 2, sd) * sqrt((nrow(X) - 1) / nrow(X))  # /n SD
  s[s == 0] <- 1
  list(Xs = sweep(sweep(X, 2, mu), 2, s, "/"), mu = mu, s = s)
}

ridge_scratch <- function(X, y, lam) {
  std <- standardize(X); Xs <- std$Xs; y_mean <- mean(y); yc <- y - y_mean
  p <- ncol(Xs)
  beta_std <- as.vector(solve(crossprod(Xs) + lam * diag(p), crossprod(Xs, yc)))
  beta <- beta_std / std$s
  list(intercept = y_mean - sum(std$mu * beta), beta = beta,
       beta_standardized = beta_std, lambda = lam)
}

soft_threshold <- function(z, t) sign(z) * pmax(abs(z) - t, 0)

lasso_scratch <- function(X, y, lam, max_iter = 1000, tol = 1e-7) {
  std <- standardize(X); Xs <- std$Xs; y_mean <- mean(y); yc <- y - y_mean
  n <- nrow(Xs); p <- ncol(Xs); beta <- rep(0, p)
  for (it in seq_len(max_iter)) {
    beta_old <- beta
    r <- yc - as.vector(Xs %*% beta)
    for (j in seq_len(p)) {
      r <- r + Xs[, j] * beta[j]
      z_j <- sum(Xs[, j] * r) / n
      beta[j] <- soft_threshold(z_j, lam / n)
      r <- r - Xs[, j] * beta[j]
    }
    if (max(abs(beta - beta_old)) < tol) break
  }
  coef <- beta / std$s
  list(intercept = y_mean - sum(std$mu * coef), beta = coef,
       beta_standardized = beta, lambda = lam,
       nonzero = sum(abs(beta) > 1e-10))
}

elastic_net_scratch <- function(X, y, lam, alpha = 0.5,
                                max_iter = 1000, tol = 1e-7) {
  std <- standardize(X); Xs <- std$Xs; y_mean <- mean(y); yc <- y - y_mean
  n <- nrow(Xs); p <- ncol(Xs); beta <- rep(0, p)
  l1 <- lam * alpha; l2 <- lam * (1 - alpha)
  for (it in seq_len(max_iter)) {
    beta_old <- beta
    r <- yc - as.vector(Xs %*% beta)
    for (j in seq_len(p)) {
      r <- r + Xs[, j] * beta[j]
      z_j <- sum(Xs[, j] * r) / n
      beta[j] <- soft_threshold(z_j, l1 / n) / (1 + l2 / n)
      r <- r - Xs[, j] * beta[j]
    }
    if (max(abs(beta - beta_old)) < tol) break
  }
  coef <- beta / std$s
  list(intercept = y_mean - sum(std$mu * coef), beta = coef,
       lambda = lam, alpha = alpha,
       nonzero = sum(abs(beta) > 1e-10))
}

# Library: glmnet::glmnet (default lasso, alpha = 1; alpha = 0 -> ridge), glmnet::cv.glmnet
library_demo <- function(X, y) {
  if (requireNamespace("glmnet", quietly = TRUE)) {
    fit <- glmnet::glmnet(X, y, alpha = 1, lambda = 0.15)   # ~ matches our lam = 15/100
    cat("glmnet lasso (alpha=1, lambda=0.15) coefs:\n"); print(as.matrix(coef(fit)))
    cv <- glmnet::cv.glmnet(X, y, alpha = 1)
    cat("\nglmnet cv.glmnet (lasso): lambda.min =", cv$lambda.min, "\n")
  } else cat("(install 'glmnet' for the canonical implementation)\n")
}

if (sys.nframe() == 0) {
  set.seed(4); n <- 100; p <- 10
  X <- matrix(rnorm(n * p), n, p)
  true_beta <- c(2, -1.5, 1, 0, 0, 0, 0.5, 0, 0, -0.3)
  y <- as.vector(X %*% true_beta) + rnorm(n)
  cat("=== Ridge (lambda = 10) ===\n");  print(ridge_scratch(X, y, 10))
  cat("\n=== Lasso (lambda = 15) ===\n"); print(lasso_scratch(X, y, 15))
  cat("\n=== Elastic net (lambda = 15, alpha = 0.5) ===\n")
  print(elastic_net_scratch(X, y, 15, alpha = 0.5))
  cat(sprintf("\n  true beta = %s\n", paste(true_beta, collapse = ", ")))
  cat("\n--- library ---\n"); library_demo(X, y)
}
