# Partial and semi-partial correlation (Reference §4.5)
# From-scratch base R plus ppcor::pcor / ppcor::spcor.
# Run with:  Rscript partial_correlation.R
#
# Inputs used below:
#   x, y         : paired numeric vectors of equal length
#   Z            : numeric vector OR matrix of covariates (n rows)
#   residualize  : "x" or "y" -- which side to residualize for semi-partial

pearson_r_scratch <- function(x, y) {
  mx <- mean(x); my <- mean(y)
  sum((x - mx) * (y - my)) / sqrt(sum((x - mx)^2) * sum((y - my)^2))
}

ols_residuals <- function(y, Z) {
  if (is.null(dim(Z))) Z <- matrix(Z, ncol = 1)
  fit <- lm(y ~ Z); resid(fit)
}

partial_correlation_scratch <- function(x, y, Z) {
  rx <- ols_residuals(x, Z); ry <- ols_residuals(y, Z)
  r <- pearson_r_scratch(rx, ry); n <- length(x)
  p <- if (is.null(dim(Z))) 1 else ncol(Z)
  df <- n - 2 - p
  t <- r * sqrt(df / (1 - r^2))
  list(partial_r = r, df = df,
       p_value = 2 * pt(-abs(t), df), n_covariates = p, method = "partial")
}

semi_partial_correlation_scratch <- function(x, y, Z, residualize = "y") {
  if (residualize == "y") { rx <- x; ry <- ols_residuals(y, Z) }
  else if (residualize == "x") { rx <- ols_residuals(x, Z); ry <- y }
  else stop("residualize must be 'x' or 'y'")
  r <- pearson_r_scratch(rx, ry); n <- length(x)
  p <- if (is.null(dim(Z))) 1 else ncol(Z)
  df <- n - 2 - p
  t <- r * sqrt(df / (1 - r^2))
  list(semi_partial_r = r, df = df,
       p_value = 2 * pt(-abs(t), df),
       residualized = residualize, method = "semi-partial")
}

partial_correlation_matrix <- function(X) {
  P <- solve(cov(X))
  d <- sqrt(diag(P)); R <- -P / outer(d, d); diag(R) <- 1; R
}

library_demo <- function(x, y, z) {
  if (requireNamespace("ppcor", quietly = TRUE)) {
    cat("ppcor::pcor.test (partial):\n"); print(ppcor::pcor.test(x, y, z))
    cat("\nppcor::spcor.test (semi-partial):\n"); print(ppcor::spcor.test(x, y, z))
  } else cat("(install 'ppcor' for pcor.test / spcor.test)\n")
}

if (sys.nframe() == 0) {
  set.seed(9); n <- 100
  z <- rnorm(n); x <- 0.7 * z + 0.3 * rnorm(n); y <- 0.8 * z + 0.2 * rnorm(n)
  cat("r_xy raw           :", pearson_r_scratch(x, y), "\n")
  cat("r_xy | z (partial) :", partial_correlation_scratch(x, y, z)$partial_r, "\n")
  cat("r_x(y.z) (semi)    :", semi_partial_correlation_scratch(x, y, z)$semi_partial_r, "\n\n")
  cat("Partial test:\n");      print(partial_correlation_scratch(x, y, z))
  cat("\nSemi-partial test:\n"); print(semi_partial_correlation_scratch(x, y, z))
  cat("\nPartial correlation matrix among (x, y, z):\n")
  print(partial_correlation_matrix(cbind(x, y, z)))
  cat("\n--- library ---\n"); library_demo(x, y, z)
}
