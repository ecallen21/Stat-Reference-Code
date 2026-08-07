# Multivariate longitudinal (Reference §12.14)
# R via nlme::lme with a stacked bivariate response (long form) or the
# joint-model packages: MCMCglmm::MCMCglmm, JM::jointModel, brms::brm.
# Run with:  Rscript multivariate_longitudinal.R

if (sys.nframe() == 0) {
  set.seed(0); N <- 100; T_ <- 5
  subject <- rep(1:N, each = T_); time <- rep(0:(T_ - 1), N)
  D <- 1^2 * matrix(c(1, 0.6, 0.6, 1), 2, 2)
  b <- MASS::mvrnorm(N, mu = c(0, 0), Sigma = D)
  Sigma <- 0.5^2 * matrix(c(1, 0.3, 0.3, 1), 2, 2)
  y1 <- numeric(N * T_); y2 <- numeric(N * T_)
  for (i in 1:N) for (j in 1:T_) {
    eps <- MASS::mvrnorm(1, c(0, 0), Sigma)
    k <- (i - 1) * T_ + j
    y1[k] <- 2 + 0.5 * time[k] + b[i, 1] + eps[1]
    y2[k] <- 1 - 0.3 * time[k] + b[i, 2] + eps[2]
  }
  # Stacked bivariate form
  long <- data.frame(
    subject = rep(subject, 2),
    time    = rep(time, 2),
    outcome = factor(rep(c("y1", "y2"), each = length(y1))),
    y       = c(y1, y2)
  )
  if (requireNamespace("nlme", quietly = TRUE)) {
    cat("=== nlme::lme stacked bivariate model ===\n")
    fit <- nlme::lme(y ~ outcome + outcome:time, random = ~ outcome | subject,
                     data = long, control = list(opt = "optim"))
    print(summary(fit))
  }
}
