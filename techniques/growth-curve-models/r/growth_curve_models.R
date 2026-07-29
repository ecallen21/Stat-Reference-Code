# Growth Curve Models via LMM (Reference §12.4)
# Base R via lme4::lmer with random intercept + slope on time.
# Run with:  Rscript growth_curve_models.R

if (sys.nframe() == 0) {
  set.seed(29); n_subj <- 40; n_time <- 6; n <- n_subj * n_time
  subject <- rep(1:n_subj, each = n_time)
  time <- rep(0:(n_time - 1), n_subj)
  u0 <- rnorm(n_subj, 0, 0.8); u1 <- rnorm(n_subj, 0, 0.2)
  y <- (50 + u0[subject]) + (2 + u1[subject]) * time + rnorm(n, 0, 0.5)
  df <- data.frame(y = y, time = time, subject = factor(subject))
  if (requireNamespace("lme4", quietly = TRUE)) {
    cat("=== Linear growth curve: lme4::lmer(y ~ time + (time | subject)) ===\n")
    print(summary(lme4::lmer(y ~ time + (time | subject), data = df, REML = TRUE)))
    cat("\n=== Quadratic growth curve ===\n")
    df$y2 <- df$y + 0.1 * time^2
    print(summary(lme4::lmer(y2 ~ time + I(time^2) + (time | subject), data = df, REML = TRUE)))
  }
}
