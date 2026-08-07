# Difference-in-Differences (Reference §15.4)
# R via lm() for the 2x2 setup, fixest::feols for TWFE, or did::att_gt for
# heterogeneous-timing Callaway-Sant'Anna.
# Run with:  Rscript diff_in_diff.R

if (sys.nframe() == 0) {
  set.seed(0); n_per <- 100; att <- 1.5
  df <- expand.grid(treated = 0:1, post = 0:1)
  df <- df[rep(1:4, each = n_per), ]
  df$y <- with(df, 2 + 0.5 * treated + 0.3 * post + att * treated * post +
                    rnorm(nrow(df)))
  cat("=== 2x2 DID via lm ===\n")
  print(summary(lm(y ~ treated * post, data = df)))

  # Staggered adoption panel
  N <- 40; T <- 6
  unit <- rep(1:N, each = T); time <- rep(1:T, N)
  treat_time <- sample(c(2, 3, 4, Inf), N, replace = TRUE)
  D <- as.integer(time >= treat_time[unit])
  alpha_i <- rnorm(N); gamma_t <- c(0, 0.1, 0.2, 0.3, 0.4, 0.5)
  y <- alpha_i[unit] + gamma_t[time] + 2 * D + rnorm(N * T, 0, 0.5)
  if (requireNamespace("fixest", quietly = TRUE)) {
    cat("\n=== fixest::feols two-way FE ===\n")
    print(fixest::feols(y ~ D | unit + time,
                        data = data.frame(unit = unit, time = time, y = y, D = D)))
  }
}
