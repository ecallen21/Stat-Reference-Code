# Marginal effects for GLMs (Reference §7.37, §7.47)
# From-scratch + library: margins::margins or marginaleffects::avg_slopes.
# Run with:  Rscript marginal_effects.R

ame_logit <- function(X, beta) {
  eta <- as.vector(X %*% beta); pi <- 1 / (1 + exp(-eta))
  mean(pi * (1 - pi)) * beta
}

mem_logit <- function(X, beta) {
  eta <- sum(colMeans(X) * beta); pi <- 1 / (1 + exp(-eta))
  pi * (1 - pi) * beta
}

ame_probit <- function(X, beta) {
  eta <- as.vector(X %*% beta); mean(dnorm(eta)) * beta
}

ame_poisson <- function(X, beta, offset = NULL) {
  eta <- as.vector(X %*% beta)
  if (!is.null(offset)) eta <- eta + offset
  mean(exp(eta)) * beta
}

discrete_ame_logit <- function(X, beta, binary_index) {
  X0 <- X; X0[, binary_index] <- 0
  X1 <- X; X1[, binary_index] <- 1
  pi0 <- 1 / (1 + exp(-as.vector(X0 %*% beta)))
  pi1 <- 1 / (1 + exp(-as.vector(X1 %*% beta)))
  mean(pi1 - pi0)
}

if (sys.nframe() == 0) {
  set.seed(10); n <- 500
  x1 <- rnorm(n); x2 <- rnorm(n); treat <- sample(0:1, n, replace = TRUE)
  eta <- -0.5 + 0.8 * x1 - 0.4 * x2 + 0.6 * treat
  y <- as.integer(runif(n) < 1 / (1 + exp(-eta)))
  X <- cbind(1, x1, x2, treat); beta <- c(-0.5, 0.8, -0.4, 0.6)
  cat("AME (logit):\n"); print(ame_logit(X, beta))
  cat("\nMEM:\n"); print(mem_logit(X, beta))
  cat("\nDiscrete AME for treat:", discrete_ame_logit(X, beta, 4), "\n")
  if (requireNamespace("margins", quietly = TRUE)) {
    fit <- glm(y ~ x1 + x2 + treat, family = binomial)
    cat("\nmargins::margins:\n"); print(summary(margins::margins(fit)))
  }
}
