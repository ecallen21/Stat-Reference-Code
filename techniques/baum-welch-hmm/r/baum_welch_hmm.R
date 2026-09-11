# Baum-Welch Algorithm (Reference Sec 47.156).
#
# Baum-Petrie 1966; Baum et al 1970. EM for HMMs (forward-backward
# posteriors + M-step).
#
# Uses `HMM::baumWelch`.

if (!requireNamespace("HMM", quietly = TRUE)) install.packages("HMM")
library(HMM)                                          # Baum-Welch EM for HMMs

set.seed(0)
hmm_true <- initHMM(States = c("A", "B"), Symbols = as.character(1:6),
                     startProbs = c(0.6, 0.4),
                     transProbs = matrix(c(0.9, 0.1, 0.2, 0.8), 2, byrow = TRUE),
                     emissionProbs = rbind(rep(1/6, 6),
                                             c(0.10, 0.10, 0.10, 0.10, 0.10, 0.50)))
sim <- simHMM(hmm_true, 800)

# Random-init HMM
hmm_init <- initHMM(States = c("A", "B"), Symbols = as.character(1:6),
                     startProbs = c(0.5, 0.5),
                     transProbs = matrix(rep(0.5, 4), 2),
                     emissionProbs = matrix(rep(1/6, 12), 2))
res <- baumWelch(hmm_init, sim$observation, maxIterations = 100)
cat("=== Baum-Welch on 2-state / 6-symbol HMM ===\n")
cat("Learned transition matrix:\n"); print(round(res$hmm$transProbs, 3))
cat("Learned emission matrix:\n"); print(round(res$hmm$emissionProbs, 3))
