# BOIN - Bayesian Optimal Interval (Reference Sec 47.263)
# Native R via BOIN / boinet / trialr; Python via from-scratch.
# Run with:  Rscript boin_bayesian_optimal_interval.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  BOIN                    -- original BOIN with GUI-driven config\n")
  cat("  boinet                  -- BOIN with efficacy-toxicity extensions\n")
  cat("  trialr::stan_boin       -- Bayesian BOIN variants\n")
  cat("  dfcrm                   -- classical CRM benchmark for comparison\n")
  cat("Python:\n")
  cat("  UBCRM / trialr (via reticulate)\n")
  cat("  U-Chicago BOIN web app for standalone use\n")
  cat("  From-scratch (see boin_bayesian_optimal_interval.py)\n")
  cat("Refs: Liu, S. & Yuan, Y. (2015) 'Bayesian optimal interval designs for\n")
  cat("      phase I clinical trials', J R Stat Soc C 64(3), 507-523.\n")
}
