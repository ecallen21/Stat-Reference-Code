# Continuation-ratio model for ordinal outcomes (Reference §8.9)
# From-scratch base R + VGAM::vglm(sratio=...) as library cross-check.
# Run with:  Rscript continuation_ratio.R
#
# Inputs:
#   X : n x p design matrix WITHOUT intercept
#   y : ordinal outcome in 1..K

fit_continuation_ratio <- function(X, y, K = NULL) {
  X <- as.matrix(X)
  if (is.null(K)) K <- max(y)
  transitions <- vector("list", K - 1)
  total_ll <- 0; total_params <- 0
  for (k in seq_len(K - 1)) {
    mask <- y >= k
    Xk <- cbind(1, X[mask, , drop = FALSE])
    yk <- as.numeric(y[mask] > k)
    m <- glm.fit(Xk, yk, family = binomial())
    ll <- sum(dbinom(yk, 1, m$fitted.values, log = TRUE))
    se <- sqrt(diag(vcov_glm_fit(Xk, m$fitted.values)))
    transitions[[k]] <- list(
      transition = sprintf("P(Y > %d | Y >= %d)", k, k),
      n_at_risk = sum(mask), n_advanced = sum(yk),
      intercept = m$coefficients[1],
      coefficients = m$coefficients[-1],
      SE = se, log_lik = ll)
    total_ll <- total_ll + ll
    total_params <- total_params + length(m$coefficients)
  }
  list(K = K, transitions = transitions,
       total_log_lik = total_ll, n_params = total_params)
}

vcov_glm_fit <- function(X, mu) {
  W <- pmax(mu * (1 - mu), 1e-12)
  MASS::ginv(t(X) %*% (X * W))
}

fit_cr_common <- function(X, y, K = NULL) {
  X <- as.matrix(X)
  if (is.null(K)) K <- max(y)
  X_stack <- NULL; y_stack <- c(); alpha_mat <- NULL
  for (k in seq_len(K - 1)) {
    mask <- y >= k
    X_stack <- rbind(X_stack, X[mask, , drop = FALSE])
    y_stack <- c(y_stack, as.numeric(y[mask] > k))
    a_row <- rep(0, K - 1); a_row[k] <- 1
    alpha_mat <- rbind(alpha_mat, matrix(rep(a_row, sum(mask)), nrow = sum(mask), byrow = TRUE))
  }
  design <- cbind(alpha_mat, X_stack)
  m <- glm.fit(design, y_stack, family = binomial(), intercept = FALSE)
  list(alpha = m$coefficients[seq_len(K - 1)],
       beta_common = m$coefficients[-seq_len(K - 1)],
       log_lik = sum(dbinom(y_stack, 1, m$fitted.values, log = TRUE)),
       n_params = length(m$coefficients))
}

lr_test_proportionality <- function(X, y, K = NULL) {
  f <- fit_continuation_ratio(X, y, K)
  r <- fit_cr_common(X, y, K)
  dll <- f$total_log_lik - r$log_lik
  ddf <- f$n_params - r$n_params
  list(LR = 2 * dll, delta_df = ddf,
       p_value = pchisq(2 * dll, ddf, lower.tail = FALSE))
}

if (sys.nframe() == 0) {
  set.seed(9); n <- 500
  x1 <- rnorm(n); x2 <- rnorm(n); X <- cbind(x1, x2)
  linpred <- 0.5 + 0.7 * x1 - 0.4 * x2
  y <- rep(1L, n)
  for (k in 1:3) {
    alpha_k <- -0.5 - 0.3 * (k - 1)
    p <- plogis(alpha_k + linpred)
    adv <- runif(n) < p
    y <- ifelse(y == k & adv, k + 1L, y)
  }
  cat("=== Category-specific CR ===\n"); print(fit_continuation_ratio(X, y, K = 4))
  cat("\n=== Proportional CR ===\n"); print(fit_cr_common(X, y, K = 4))
  cat("\n=== LR test ===\n"); print(lr_test_proportionality(X, y, K = 4))

  if (requireNamespace("VGAM", quietly = TRUE)) {
    cat("\n--- library: VGAM::vglm(sratio) ---\n")
    df <- data.frame(y = ordered(y), x1 = x1, x2 = x2)
    print(coef(VGAM::vglm(y ~ x1 + x2, family = VGAM::sratio(parallel = TRUE), data = df)))
  }
}
