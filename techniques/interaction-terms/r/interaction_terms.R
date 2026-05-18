# Interaction terms in regression (Reference §5.16, §5.24, §5.25, §5.37)
# From-scratch base R plus stats::lm (which handles `:` and `*` operators).
# Run with:  Rscript interaction_terms.R
#
# Inputs used below:
#   x1, x2  : continuous predictors
#   x, group: continuous + categorical
#   center  : if TRUE, center x1 and x2 before forming the product

fit_ols_with_se <- function(X, y) {
  X <- as.matrix(X); y <- as.numeric(y); n <- nrow(X); p <- ncol(X)
  beta <- as.vector(solve(crossprod(X), crossprod(X, y)))
  e <- y - as.vector(X %*% beta); rss <- sum(e^2); sigma2 <- rss / (n - p)
  var_beta <- sigma2 * solve(crossprod(X)); se <- sqrt(diag(var_beta))
  t_stats <- beta / se; p_vals <- 2 * pt(-abs(t_stats), n - p)
  list(beta = beta, se = se, t = t_stats, p = p_vals,
       rss = rss, sigma_hat = sqrt(sigma2), df_residual = n - p)
}

continuous_by_continuous_scratch <- function(x1, x2, y, center = FALSE) {
  if (center) { x1 <- x1 - mean(x1); x2 <- x2 - mean(x2) }
  X <- cbind(1, x1, x2, x1 * x2)
  res <- fit_ols_with_se(X, y)
  res$names <- c("intercept", "x1", "x2", "x1:x2"); res$centered <- center; res
}

categorical_by_continuous_scratch <- function(x, group, y) {
  levels <- sort(unique(group)); ref <- levels[1]
  D <- if (length(levels) > 1)
    sapply(levels[-1], function(lev) as.numeric(group == lev)) else matrix(0, length(x), 0)
  if (is.null(dim(D))) D <- matrix(D, ncol = 1)
  inter <- if (ncol(D) > 0) sweep(D, 1, x, "*") else matrix(0, length(x), 0)
  X <- cbind(1, x, D, inter)
  res <- fit_ols_with_se(X, y)
  res$names <- c("intercept", "x",
                 paste0("group[", levels[-1], "]"),
                 paste0("x:group[", levels[-1], "]"))
  slopes <- list(); slopes[[ref]] <- res$beta[2]
  for (k in seq_along(levels[-1]))
    slopes[[levels[-1][k]]] <- res$beta[2] + res$beta[2 + length(levels[-1]) + k]
  res$simple_slopes <- slopes; res
}

marginal_effect_x1 <- function(beta, x2_values) beta[2] + beta[4] * x2_values

# Library: lm(y ~ x1 * x2)  is the idiomatic call; * expands to main effects + interaction
library_demo <- function(x1, x2, y, x, group, y2) {
  cat("lm(y ~ x1 * x2)  uncentered:\n");  print(summary(lm(y ~ x1 * x2)))
  cat("\nlm(y ~ I(x1 - mean(x1)) * I(x2 - mean(x2))):\n")
  print(summary(lm(y ~ I(x1 - mean(x1)) * I(x2 - mean(x2)))))
  cat("\nlm(y2 ~ x * group):\n"); print(summary(lm(y2 ~ x * group)))
}

if (sys.nframe() == 0) {
  set.seed(3); n <- 200
  x1 <- rnorm(n, 10, 3); x2 <- rnorm(n, 5, 2)
  y <- 1 + 0.5 * x1 - 0.3 * x2 + 0.1 * x1 * x2 + rnorm(n)
  cat("=== Continuous x continuous (uncentered) ===\n")
  raw <- continuous_by_continuous_scratch(x1, x2, y, center = FALSE)
  for (i in seq_along(raw$beta))
    cat(sprintf("  %-10s beta = %+.4f  SE = %.4f  p = %.4g\n",
                raw$names[i], raw$beta[i], raw$se[i], raw$p[i]))
  cat("\n=== Continuous x continuous (CENTERED, recommended) ===\n")
  cen <- continuous_by_continuous_scratch(x1, x2, y, center = TRUE)
  for (i in seq_along(cen$beta))
    cat(sprintf("  %-10s beta = %+.4f  SE = %.4f  p = %.4g\n",
                cen$names[i], cen$beta[i], cen$se[i], cen$p[i]))

  group <- sample(c("A", "B", "C"), n, replace = TRUE)
  slope_by_group <- c(A = 0.2, B = 0.6, C = 1.0)
  y2 <- 1 + slope_by_group[group] * x1 + rnorm(n)
  cat("\n=== Categorical (3 levels) by continuous ===\n")
  res <- categorical_by_continuous_scratch(x1, group, y2)
  for (i in seq_along(res$beta))
    cat(sprintf("  %-18s beta = %+.4f  SE = %.4f  p = %.4g\n",
                res$names[i], res$beta[i], res$se[i], res$p[i]))
  cat("\n  simple slopes:\n"); print(res$simple_slopes)

  cat("\n--- library ---\n"); library_demo(x1, x2, y, x1, group, y2)
}
