# Viterbi Algorithm (Reference Sec 47.155).
#
# Viterbi 1967. Dynamic programming for MAP decoding of HMMs.
#
# Uses `HMM::viterbi` (R HMM package).

if (!requireNamespace("HMM", quietly = TRUE)) install.packages("HMM")
library(HMM)                                          # HMM inference + Viterbi

set.seed(0)
hmm <- initHMM(States = c("Fair", "Loaded"),
                Symbols = as.character(1:6),
                startProbs = c(0.5, 0.5),
                transProbs = matrix(c(0.95, 0.05, 0.10, 0.90), 2, byrow = TRUE),
                emissionProbs = rbind(rep(1/6, 6),
                                        c(0.10, 0.10, 0.10, 0.10, 0.10, 0.50)))
sim <- simHMM(hmm, 300)
path <- viterbi(hmm, sim$observation)
acc <- mean(path == sim$states)
cat("=== Viterbi on fair/loaded die HMM (Viterbi 1967) ===\n")
cat(sprintf("  T = 300, Viterbi decoding accuracy = %.3f\n", acc))
