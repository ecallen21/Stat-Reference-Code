# Nonlinear Mixed-Effects Models (Reference §12.12)
# Base R via nlme::nlme (authoritative joint MLE) + nls per subject.
# Run with:  Rscript nonlinear_mixed_effects.R

if (sys.nframe() == 0) {
  set.seed(43); n_subj <- 40; n_time <- 8; n <- n_subj * n_time
  subject <- rep(1:n_subj, each = n_time)
  time <- rep(seq(0, 10, length.out = n_time), n_subj)
  true_pop <- c(100, 0.6, 5)
  D <- diag(c(100, 0.02, 0.5))
  b <- MASS::mvrnorm(n_subj, c(0, 0, 0), D)
  y <- numeric(n)
  for (i in 1:n_subj) {
    m <- subject == i; theta <- true_pop + b[i, ]
    y[m] <- theta[1] / (1 + exp(-theta[2] * (time[m] - theta[3]))) + rnorm(sum(m), 0, 2)
  }
  df <- data.frame(y = y, time = time, subject = factor(subject))
  if (requireNamespace("nlme", quietly = TRUE)) {
    cat("=== nlme::nlme (joint NLME MLE) ===\n")
    fit <- nlme::nlme(y ~ A / (1 + exp(-r * (time - m))),
                       data = df, fixed = A + r + m ~ 1,
                       random = A + r + m ~ 1 | subject,
                       start = c(A = 80, r = 0.5, m = 4))
    print(summary(fit))
  }
}
