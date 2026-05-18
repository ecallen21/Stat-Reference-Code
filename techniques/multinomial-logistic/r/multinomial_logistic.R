# Multinomial logistic regression (Reference §7.4)
# From-scratch via optim plus nnet::multinom (canonical) or VGAM::vglm.
# Run with:  Rscript multinomial_logistic.R

softmax <- function(eta_full) {
  e <- exp(eta_full - apply(eta_full, 1, max))
  e / rowSums(e)
}

neg_log_lik <- function(theta, X, y, K) {
  n <- nrow(X); p <- ncol(X)
  B <- matrix(theta, K - 1, p, byrow = FALSE)
  eta <- X %*% t(B); eta_full <- cbind(eta, 0)
  P <- softmax(eta_full)
  -sum(log(pmax(P[cbind(seq_len(n), y)], 1e-15)))
}

fit_scratch <- function(X, y) {
  X <- as.matrix(X); y <- as.integer(y); n <- nrow(X); p <- ncol(X); K <- max(y)
  theta0 <- rep(0, (K - 1) * p)
  res <- optim(theta0, neg_log_lik, X = X, y = y, K = K, method = "BFGS",
               hessian = TRUE, control = list(reltol = 1e-9))
  B <- matrix(res$par, K - 1, p, byrow = FALSE)
  H_inv <- solve(res$hessian)
  se <- matrix(sqrt(diag(H_inv)), K - 1, p, byrow = FALSE)
  zc <- qnorm(0.975)
  list(reference_class = K, beta = B, se = se, z = B / se,
       p_values = 2 * pnorm(-abs(B / se)),
       odds_ratio_vs_ref = exp(B),
       or_ci_lower = exp(B - zc * se), or_ci_upper = exp(B + zc * se),
       log_likelihood = -res$value,
       converged = (res$convergence == 0))
}

library_demo <- function(X, y) {
  if (requireNamespace("nnet", quietly = TRUE)) {
    df <- data.frame(y = factor(y), X)
    print(summary(nnet::multinom(y ~ ., data = df, trace = FALSE)))
  } else cat("(install 'nnet' for multinom)\n")
}

if (sys.nframe() == 0) {
  set.seed(2); n <- 400
  x1 <- rnorm(n); x2 <- rnorm(n); X <- cbind(1, x1, x2)
  B_true <- matrix(c(0.5, 1, -0.5, -0.3, -0.6, 0.8), nrow = 2, byrow = TRUE)
  eta <- X %*% t(B_true); eta_full <- cbind(eta, 0)
  P <- softmax(eta_full)
  y <- apply(P, 1, function(p) sample(1:3, 1, prob = p))
  res <- fit_scratch(X, y); print(res)
  cat("\n--- library ---\n"); library_demo(X[, -1], y)
}
