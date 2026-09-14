# Wald SPRT (Wald 1945)
# R: `Sequential::CV.G.Bern`, `gsDesign::sprt`,
#    `sequential::LR` (varying package APIs);
#    also base R custom loop is common.
# Python: `statsmodels.stats.sequential`, from-scratch
#
# alpha <- beta <- 0.05
# logA <- log((1 - beta) / alpha); logB <- log(beta / (1 - alpha))
# lr <- 0
# for (x in stream) {
#   lr <- lr + log(dbinom(x, 1, 0.6) / dbinom(x, 1, 0.4))
#   if (lr >= logA) return("H1")
#   if (lr <= logB) return("H0")
# }
