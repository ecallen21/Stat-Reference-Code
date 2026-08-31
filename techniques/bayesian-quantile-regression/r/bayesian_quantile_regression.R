# Bayesian quantile regression (Reference Sec 33.2)
# Native R via bayesQR / brms; Python via reticulate.
# Run with:  Rscript bayesian_quantile_regression.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  bayesQR                     -- Kotz-Yu asymmetric Laplace MCMC for QR\n")
  cat("  brms (family = 'asym_laplace') -- Stan backend, flexible priors + hierarchical\n")
  cat("  quantreg                    -- frequentist QR (baseline comparison)\n")
  cat("Python:\n")
  cat("  pymc (asymmetric Laplace + Normal prior) -- probabilistic QR\n")
  cat("  numpyro                      -- HMC-based Bayesian QR\n")
  cat("  statsmodels.QuantReg         -- frequentist comparison\n")
  cat("Refs: Yu, K. & Moyeed, R. (2001) 'Bayesian quantile regression', Stat & Prob Lett.\n")
  cat("      Koenker, R. (2005) 'Quantile Regression', Cambridge U.P.\n")
}
