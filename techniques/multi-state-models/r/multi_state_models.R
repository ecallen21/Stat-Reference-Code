# Multi-state models: illness-death (Reference §11.27, §11.52)
# Base R via mstate + survival packages (authoritative implementations).
# Run with:  Rscript multi_state_models.R

if (sys.nframe() == 0) {
  cat("For illness-death multi-state models in R:\n\n")
  cat("  library(mstate)\n")
  cat("  tmat <- transMat(list(c(2, 3), c(3), c()))  # 3 states: healthy, ill, dead\n")
  cat("  msm_data <- msprep(...)\n")
  cat("  fit <- coxph(Surv(Tstart, Tstop, status) ~ ... + strata(trans), data = msm_data)\n")
  cat("  # State-occupation probabilities via probtrans().\n\n")
  cat("Also see msm::msm() for time-homogeneous Markov models on interval-censored data.\n")
}
