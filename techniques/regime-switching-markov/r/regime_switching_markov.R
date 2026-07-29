# Markov-switching model (Reference §13.14, §13.15)
# R via MSwM::msmFit or depmixS4::depmix.
# Run with:  Rscript regime_switching_markov.R

if (sys.nframe() == 0) {
  set.seed(2); T_ <- 500
  S <- integer(T_); P_true <- rbind(c(0.95, 0.05), c(0.10, 0.90))
  for (t in 2:T_) S[t] <- sample(0:1, 1, prob = P_true[S[t - 1] + 1, ])
  y <- ifelse(S == 0, rnorm(T_, 0, 0.5), rnorm(T_, 1, 2.5))

  if (requireNamespace("depmixS4", quietly = TRUE)) {
    cat("=== depmixS4::depmix (K = 2 Gaussian) ===\n")
    mod <- depmixS4::depmix(y ~ 1, data = data.frame(y = y), nstates = 2)
    fit <- depmixS4::fit(mod, verbose = FALSE)
    print(summary(fit))
  } else if (requireNamespace("MSwM", quietly = TRUE)) {
    cat("=== MSwM::msmFit on lm(y ~ 1) ===\n")
    lm0 <- lm(y ~ 1)
    print(MSwM::msmFit(lm0, k = 2, sw = c(TRUE, TRUE)))
  }
}
