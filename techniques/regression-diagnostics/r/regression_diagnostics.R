# Regression diagnostics after OLS (Reference §5.6, §5.30, §5.39)
# Run with:  Rscript regression_diagnostics.R
#
# Inputs used below:
#   X  : n x p design matrix (first column = 1s for the intercept)
#   y  : response vector of length n

hat_matrix <- function(X) X %*% solve(crossprod(X)) %*% t(X)

diagnostics_scratch <- function(X, y) {
  X <- as.matrix(X); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  beta <- solve(crossprod(X), crossprod(X, y))
  yhat <- as.vector(X %*% beta); e <- y - yhat
  H <- hat_matrix(X); h <- diag(H)
  rss <- sum(e^2); sigma2 <- rss / (n - p); sigma <- sqrt(sigma2)
  one_minus_h <- pmax(1 - h, 1e-15)
  s <- e / (sigma * sqrt(one_minus_h))                         # standardized
  sigma2_i <- ((n - p) * sigma2 - e^2 / one_minus_h) / (n - p - 1)
  sigma_i <- sqrt(pmax(sigma2_i, 0))
  t_i <- e / (sigma_i * sqrt(one_minus_h))                     # studentized (deleted)
  cooks <- (s^2 / p) * (h / one_minus_h)
  dffits <- t_i * sqrt(h / one_minus_h)
  XtX_inv <- solve(crossprod(X))
  R <- XtX_inv %*% t(X)                                        # p x n
  delta_beta <- R * (e / one_minus_h)[col(R)]                  # p x n  (row by row)
  diagXtX <- sqrt(diag(XtX_inv))
  dfbetas <- t(delta_beta) / outer(sigma_i, diagXtX)           # n x p
  list(n = n, p = p,
       leverage = h, residuals = e, std_residuals = s, studentized = t_i,
       cooks_d = cooks, dffits = dffits, dfbetas = dfbetas,
       sigma_hat = sigma, df_residual = n - p,
       thresholds = list(
         leverage_high = 2 * p / n, cooks_d_large = 4 / n,
         dffits_large = 2 * sqrt(p / n), dfbetas_large = 2 / sqrt(n)))
}

# Library: stats::hatvalues, cooks.distance, dffits, dfbetas, rstandard, rstudent
library_demo <- function(df) {
  fit <- lm(y ~ x1 + x2, data = df)
  cat("hatvalues (first 5):", head(hatvalues(fit), 5), "\n")
  cat("cooks.distance (first 5):", head(cooks.distance(fit), 5), "\n")
  cat("dffits (first 5):", head(dffits(fit), 5), "\n")
  cat("rstudent (first 5):", head(rstudent(fit), 5), "\n")
}

if (sys.nframe() == 0) {
  set.seed(0); n <- 30
  x1 <- rnorm(n); x2 <- rnorm(n)
  y <- 1 + 2 * x1 - 1.5 * x2 + rnorm(n)
  x1[1] <- 5; y[1] <- -10           # influential observation
  X <- cbind(1, x1, x2)
  d <- diagnostics_scratch(X, y)
  cat("=== First 6 observations ===\n")
  cat(sprintf("  %3s %8s %8s %8s %8s %8s\n", "i", "lev", "std_r", "stud", "cookD", "dffits"))
  for (i in 1:6)
    cat(sprintf("  %3d %8.4f %8.3f %8.3f %8.4f %8.3f\n",
                i, d$leverage[i], d$std_residuals[i], d$studentized[i],
                d$cooks_d[i], d$dffits[i]))
  cat("\nThresholds:\n"); print(d$thresholds)
  flagged <- which(d$leverage > d$thresholds$leverage_high |
                   d$cooks_d > d$thresholds$cooks_d_large)
  cat("\nFlagged indices:", flagged, "\n")
  cat("\n--- library ---\n"); library_demo(data.frame(y = y, x1 = x1, x2 = x2))
}
