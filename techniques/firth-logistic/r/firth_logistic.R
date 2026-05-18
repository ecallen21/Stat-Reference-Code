# Firth's penalized logistic regression (Reference §7.7, §7.51)
# From-scratch + logistf::logistf (the canonical R implementation).
# Run with:  Rscript firth_logistic.R

logistic_pi <- function(eta) 1 / (1 + exp(-pmin(pmax(eta, -500), 500)))

firth_fit_scratch <- function(X, y, max_iter = 100, tol = 1e-7) {
  X <- as.matrix(X); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  beta <- rep(0, p); converged <- FALSE
  for (it in seq_len(max_iter)) {
    eta <- as.vector(X %*% beta); pi <- logistic_pi(eta)
    w <- pmax(pi * (1 - pi), 1e-12)
    XtWX_inv <- solve(crossprod(X * sqrt(w)))
    h <- rowSums((X %*% XtWX_inv) * X) * w
    y_adj <- y + h * (0.5 - pi)
    z <- eta + (y_adj - pi) / w
    sw <- sqrt(w); Xw <- sw * X; zw <- sw * z
    beta_new <- as.vector(solve(crossprod(Xw), crossprod(Xw, zw)))
    if (max(abs(beta_new - beta)) < tol) {
      beta <- beta_new; converged <- TRUE; break
    }
    beta <- beta_new
  }
  eta <- as.vector(X %*% beta); pi <- logistic_pi(eta)
  w <- pmax(pi * (1 - pi), 1e-12)
  var_beta <- solve(crossprod(X * sqrt(w))); se <- sqrt(diag(var_beta))
  list(beta = beta, se = se, z = beta / se,
       p_values = 2 * pnorm(-abs(beta / se)),
       odds_ratio = exp(beta), converged = converged, iterations = it)
}

if (sys.nframe() == 0) {
  set.seed(7); n <- 50
  x1 <- rnorm(n); x2 <- rnorm(n)
  y <- as.integer(x1 > 0.5)
  X <- cbind(1, x1, x2)
  cat("=== Standard glm (likely warns about separation) ===\n")
  fit_std <- tryCatch(glm(y ~ x1 + x2, family = binomial), error = function(e) e)
  print(coef(fit_std))
  cat("\n=== Firth (from scratch) ===\n"); print(firth_fit_scratch(X, y))
  cat("\n=== logistf (canonical Firth) ===\n")
  if (requireNamespace("logistf", quietly = TRUE))
    print(logistf::logistf(y ~ x1 + x2))
  else cat("(install 'logistf' for the canonical Firth implementation)\n")
}
