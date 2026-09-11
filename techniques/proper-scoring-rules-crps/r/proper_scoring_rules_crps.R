# Proper Scoring Rules - CRPS (Reference Sec 47.255)
# Native R via scoringRules / scoringutils / verification; Python via properscoring / from-scratch.
# Run with:  Rscript proper_scoring_rules_crps.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  scoringRules::crps         -- closed-form CRPS for many families\n")
  cat("  scoringutils::score        -- forecast-evaluation toolkit\n")
  cat("  verification::crps         -- weather / climate forecasting\n")
  cat("  ensembleBMA::crps          -- Bayesian Model Averaging ensembles\n")
  cat("Python:\n")
  cat("  properscoring.crps_ensemble / crps_gaussian\n")
  cat("  scipy: log_score, brier_score_loss (sklearn)\n")
  cat("  From-scratch (see proper_scoring_rules_crps.py)\n")
  cat("Refs: Gneiting, T. & Raftery, A.E. (2007) 'Strictly Proper Scoring\n")
  cat("      Rules, Prediction, and Estimation', JASA 102(477), 359-378.\n")
}
