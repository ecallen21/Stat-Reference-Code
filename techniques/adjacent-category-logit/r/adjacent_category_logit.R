# Adjacent-category logit for ordinal outcomes (Reference §8.10)
# From-scratch base R (BFGS on the multinomial log-likelihood) plus
# VGAM::vglm(acat=...) as library cross-check.
# Run with:  Rscript adjacent_category_logit.R
#
# Sign convention: positive beta => positive X shifts toward HIGHER category.

softmax_neg_ll <- function(theta, X, y, K) {
  n <- nrow(X); p <- ncol(X)
  gamma <- theta[seq_len(K - 1)]
  beta <- theta[K : (K - 1 + p)]
  scale <- (K - 1):1
  eta <- outer(as.vector(X %*% beta), -scale) + matrix(gamma, n, K - 1, byrow = TRUE)
  eta <- cbind(eta, 0)
  m <- apply(eta, 1, max)
  lse <- m + log(rowSums(exp(eta - m)))
  ll <- sum(eta[cbind(seq_len(n), y)] - lse)
  -ll
}

fit_ac_common <- function(X, y, K = NULL) {
  X <- as.matrix(X); if (is.null(K)) K <- max(y)
  p <- ncol(X)
  theta0 <- rep(0, (K - 1) + p)
  res <- optim(theta0, softmax_neg_ll, X = X, y = y, K = K,
                method = "BFGS", hessian = TRUE)
  cov <- tryCatch(solve(res$hessian), error = function(e) matrix(NA, length(res$par), length(res$par)))
  se <- sqrt(pmax(diag(cov), 0))
  list(gamma = res$par[seq_len(K - 1)],
       beta_common = res$par[K:(K - 1 + p)],
       SE_gamma = se[seq_len(K - 1)],
       SE_beta = se[K:(K - 1 + p)],
       log_lik = -res$value,
       K = K)
}

fit_ac_pairs <- function(X, y, K = NULL) {
  X <- as.matrix(X); if (is.null(K)) K <- max(y)
  out <- list()
  total_ll <- 0
  for (k in seq_len(K - 1)) {
    mask <- (y == k) | (y == k + 1)
    Xk <- cbind(1, X[mask, , drop = FALSE])
    yk <- as.numeric(y[mask] == k + 1)
    m <- glm.fit(Xk, yk, family = binomial(), intercept = FALSE)
    ll <- sum(dbinom(yk, 1, m$fitted.values, log = TRUE))
    out[[k]] <- list(pair = sprintf("%d vs %d", k, k + 1),
                     intercept = m$coefficients[1],
                     beta_k = m$coefficients[-1],
                     n = sum(mask), log_lik = ll)
    total_ll <- total_ll + ll
  }
  list(pairs = out, total_log_lik = total_ll)
}

if (sys.nframe() == 0) {
  set.seed(12); n <- 600
  x1 <- rnorm(n); x2 <- rnorm(n); X <- cbind(x1, x2)
  beta_true <- c(0.5, -0.3); gamma_true <- c(-1, -1.5, -2)
  K <- 4; scale <- (K - 1):1
  eta <- outer(as.vector(X %*% beta_true), -scale) + matrix(gamma_true, n, K - 1, byrow = TRUE)
  eta <- cbind(eta, 0)
  probs <- exp(eta - apply(eta, 1, max)); probs <- probs / rowSums(probs)
  y <- apply(probs, 1, function(pr) sample.int(K, 1, prob = pr))

  cat("=== Common-beta AC (target beta = 0.5, -0.3) ===\n"); print(fit_ac_common(X, y, K))
  cat("\n=== Pairwise AC ===\n"); print(fit_ac_pairs(X, y, K))

  if (requireNamespace("VGAM", quietly = TRUE)) {
    cat("\n--- library: VGAM::vglm(acat) ---\n")
    df <- data.frame(y = ordered(y), x1 = x1, x2 = x2)
    print(coef(VGAM::vglm(y ~ x1 + x2,
                          family = VGAM::acat(parallel = TRUE, reverse = TRUE), data = df)))
  }
}
