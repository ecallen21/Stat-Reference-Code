# Specification and diagnostic tests for OLS regression (Reference §5.7, §5.21)
# From-scratch base R plus lmtest / car package calls.
# Run with:  Rscript specification_tests.R
#
# Inputs used below:
#   X  : n x p design matrix (first column = 1s for the intercept)
#   y  : response vector

.ols <- function(X, y) {
  X <- as.matrix(X); y <- as.numeric(y)
  beta <- as.vector(solve(crossprod(X), crossprod(X, y)))
  yhat <- as.vector(X %*% beta); list(beta = beta, yhat = yhat, resid = y - yhat)
}

breusch_pagan_scratch <- function(X, y) {
  fit <- .ols(X, y); n <- nrow(X); p <- ncol(X)
  z <- fit$resid^2
  z_hat <- as.vector(X %*% solve(crossprod(X), crossprod(X, z)))
  r2 <- 1 - sum((z - z_hat)^2) / sum((z - mean(z))^2)
  LM <- n * r2; df <- p - 1
  list(LM = LM, df = df, p_value = pchisq(LM, df, lower.tail = FALSE),
       method = "Breusch-Pagan (LM)")
}

whites_test_scratch <- function(X, y) {
  X <- as.matrix(X); n <- nrow(X)
  has_int <- isTRUE(all.equal(X[, 1], rep(1, n)))
  Xc <- if (has_int) X[, -1, drop = FALSE] else X
  cols <- if (has_int) list(rep(1, n)) else list()
  for (j in seq_len(ncol(Xc))) cols[[length(cols) + 1]] <- Xc[, j]
  for (j in seq_len(ncol(Xc))) cols[[length(cols) + 1]] <- Xc[, j]^2
  if (ncol(Xc) > 1) for (i in seq_len(ncol(Xc) - 1))
    for (j in (i + 1):ncol(Xc))
      cols[[length(cols) + 1]] <- Xc[, i] * Xc[, j]
  Z <- do.call(cbind, cols)
  z <- .ols(X, y)$resid^2
  z_hat <- as.vector(Z %*% solve(crossprod(Z), crossprod(Z, z)))
  r2 <- 1 - sum((z - z_hat)^2) / sum((z - mean(z))^2)
  LM <- n * r2; df <- ncol(Z) - 1
  list(LM = LM, df = df, p_value = pchisq(LM, df, lower.tail = FALSE),
       method = "White")
}

durbin_watson_scratch <- function(X, y) {
  e <- .ols(X, y)$resid
  dw <- sum(diff(e)^2) / sum(e^2)
  interp <- if (dw < 1.5) "positive autocorrelation"
            else if (dw > 2.5) "negative autocorrelation"
            else "no strong evidence of autocorrelation"
  list(DW = dw, interpretation = interp,
       method = "Durbin-Watson (no p-value; use lmtest::dwtest)")
}

ramsey_reset_scratch <- function(X, y, powers = 2) {
  fit0 <- .ols(X, y); n <- nrow(X); p <- ncol(X)
  extras <- do.call(cbind, lapply(powers, function(k) fit0$yhat^k))
  X_aug <- cbind(X, extras); fit1 <- .ols(X_aug, y)
  rss0 <- sum(fit0$resid^2); rss1 <- sum(fit1$resid^2)
  q <- ncol(extras)
  F <- ((rss0 - rss1) / q) / (rss1 / (n - p - q))
  list(F = F, df1 = q, df2 = n - p - q,
       p_value = pf(F, q, n - p - q, lower.tail = FALSE),
       method = "Ramsey RESET")
}

run_all <- function(X, y) list(
  breusch_pagan = breusch_pagan_scratch(X, y),
  white         = whites_test_scratch(X, y),
  durbin_watson = durbin_watson_scratch(X, y),
  ramsey_reset  = ramsey_reset_scratch(X, y),
  shapiro_residuals = shapiro.test(.ols(X, y)$resid)
)

# Library: lmtest::bptest, lmtest::dwtest, lmtest::resettest;  car::ncvTest, car::resettest
library_demo <- function(df) {
  fit <- lm(y ~ x1 + x2, data = df)
  if (requireNamespace("lmtest", quietly = TRUE)) {
    cat("lmtest::bptest:\n");    print(lmtest::bptest(fit))
    cat("\nlmtest::dwtest:\n");  print(lmtest::dwtest(fit))
    cat("\nlmtest::resettest:\n"); print(lmtest::resettest(fit, power = 2))
  } else cat("(install 'lmtest' for bptest / dwtest / resettest)\n")
}

if (sys.nframe() == 0) {
  set.seed(1); n <- 100
  x1 <- rnorm(n); x2 <- rnorm(n); X <- cbind(1, x1, x2)
  cat("=== Clean ===\n");                print(run_all(X, 1 + 2 * x1 - x2 + rnorm(n)))
  cat("\n=== Heteroscedastic ===\n");    print(run_all(X, 1 + 2 * x1 - x2 + rnorm(n, sd = 1 + 0.5 * abs(x1))))
  cat("\n=== Omitted nonlinearity ===\n"); print(run_all(X, 1 + 2 * x1 + 0.7 * x1^2 - x2 + rnorm(n)))
  cat("\n--- library ---\n"); library_demo(data.frame(y = 1 + 2 * x1 + 0.7 * x1^2 - x2 + rnorm(n),
                                                       x1 = x1, x2 = x2))
}
