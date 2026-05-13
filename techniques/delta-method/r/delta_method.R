# The Delta Method (Reference §3.29)
# Run with:  Rscript delta_method.R
#
# Inputs used below:
#   g       : a function of theta (numeric vector) returning a scalar
#   theta   : the MLE / point estimate
#   V       : covariance matrix of theta
#   step    : finite-difference step for the numerical gradient (default 1e-5)

numerical_gradient <- function(g, theta, step = 1e-5) {
  theta <- as.numeric(theta); p <- length(theta)
  grad <- numeric(p)
  for (i in seq_len(p)) {
    e <- numeric(p); e[i] <- step
    grad[i] <- (g(theta + e) - g(theta - e)) / (2 * step)
  }
  grad
}

delta_se <- function(g, theta, V, step = 1e-5) {
  grad <- numerical_gradient(g, theta, step)
  sqrt(as.numeric(t(grad) %*% V %*% grad))
}

delta_ci <- function(g, theta, V, conf = 0.95, step = 1e-5) {
  est <- g(theta); se <- delta_se(g, theta, V, step)
  z <- qnorm(0.5 + conf / 2)
  list(estimate = est, se = se,
       ci_lower = est - z * se, ci_upper = est + z * se)
}

# Worked example: log OR for a 2x2 table (Poisson approx variance for cells)
log_or_ci_closed <- function(a, b, c, d, conf = 0.95) {
  log_or <- log((a * d) / (b * c))
  se <- sqrt(1/a + 1/b + 1/c + 1/d); z <- qnorm(0.5 + conf / 2)
  list(log_or = log_or, se_closed = se,
       ci_lower_log = log_or - z * se, ci_upper_log = log_or + z * se,
       ci_lower_or = exp(log_or - z * se), ci_upper_or = exp(log_or + z * se))
}

log_or_ci_delta <- function(a, b, c, d, conf = 0.95) {
  theta <- c(a, b, c, d); V <- diag(theta)
  g <- function(t) log((t[1] * t[4]) / (t[2] * t[3]))
  delta_ci(g, theta, V, conf)
}

# CV example: SE of CV = sd / mean
cv_se_delta <- function(mean_, sd_, var_mean, var_sd, cov_mean_sd = 0) {
  theta <- c(mean_, sd_); V <- matrix(c(var_mean, cov_mean_sd,
                                        cov_mean_sd, var_sd), nrow = 2)
  delta_se(function(t) t[2] / t[1], theta, V)
}

# Library: car::deltaMethod (also msm::deltamethod, marginaleffects::hypotheses)
library_demo <- function() {
  if (requireNamespace("car", quietly = TRUE)) {
    # Pseudo-fit so we can use car::deltaMethod the way you would in practice
    fit <- lm(mpg ~ wt, data = mtcars)
    cat("car::deltaMethod on lm(mpg ~ wt) -- ratio of intercept to slope:\n")
    print(car::deltaMethod(fit, "`(Intercept)` / wt"))
  } else cat("(install 'car' for deltaMethod)\n")
}

if (sys.nframe() == 0) {
  cat("=== Log-OR CI for 2x2 = [[40, 20], [10, 30]] ===\n")
  cat("Closed-form:\n");  print(log_or_ci_closed(40, 20, 10, 30))
  cat("\nDelta method (numerical grad):\n"); print(log_or_ci_delta(40, 20, 10, 30))
  cat("\n=== SE of CV = sd / mean (n = 50, mean = 50, sd = 12) ===\n")
  n <- 50; mean_ <- 50; sd_ <- 12
  se_cv <- cv_se_delta(mean_, sd_, sd_^2 / n, sd_^2 / (2 * (n - 1)))
  cat(sprintf("  CV = %.4f    SE(CV) approx = %.4f\n", sd_ / mean_, se_cv))
  cat("\n--- library ---\n"); library_demo()
}
