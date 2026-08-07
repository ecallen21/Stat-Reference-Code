# Hidden Markov Model (Reference §13.14)
# R via depmixS4 or HMM package.
# Run with:  Rscript hmm.R

if (sys.nframe() == 0) {
  set.seed(0)
  A_true <- rbind(c(0.92, 0.08), c(0.10, 0.90))
  B_true <- rbind(c(0.7, 0.2, 0.1), c(0.1, 0.2, 0.7))
  T_ <- 500
  S <- integer(T_); S[1] <- sample(1:2, 1, prob = c(0.5, 0.5))
  y <- integer(T_); y[1] <- sample(1:3, 1, prob = B_true[S[1], ])
  for (t in 2:T_) {
    S[t] <- sample(1:2, 1, prob = A_true[S[t - 1], ])
    y[t] <- sample(1:3, 1, prob = B_true[S[t], ])
  }
  if (requireNamespace("HMM", quietly = TRUE)) {
    cat("=== HMM::baumWelch ===\n")
    hmm <- HMM::initHMM(States = c("A", "B"), Symbols = as.character(1:3),
                        startProbs = c(0.5, 0.5),
                        transProbs = matrix(0.5, 2, 2),
                        emissionProbs = matrix(1/3, 2, 3))
    bw <- HMM::baumWelch(hmm, as.character(y), maxIterations = 200)
    print(bw$hmm$transProbs)
    print(bw$hmm$emissionProbs)
  } else if (requireNamespace("depmixS4", quietly = TRUE)) {
    cat("=== depmixS4::depmix categorical HMM ===\n")
    mod <- depmixS4::depmix(y ~ 1, family = multinomial(),
                             data = data.frame(y = factor(y)), nstates = 2)
    fit <- depmixS4::fit(mod, verbose = FALSE)
    print(summary(fit))
  }
}
