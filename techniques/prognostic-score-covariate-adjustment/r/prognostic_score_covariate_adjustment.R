# Prognostic-score covariate adjustment / PROCOVA (Reference Sec 44.15)
# Native R via procova / RATES; Python via unlearn-ai procova.
# Run with:  Rscript prognostic_score_covariate_adjustment.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  RATES                          -- Schuler PROCOVA implementation (Unlearn.AI)\n")
  cat("  MatchIt + prognosticscores     -- Hansen (2008) matching-based scores\n")
  cat("  optmatch                       -- prognostic-score matching\n")
  cat("  survival, rms                   -- fit m_hat via Cox / logistic / OLS\n")
  cat("Python:\n")
  cat("  procova (github.com/unlearn-ai/procova)  -- Schuler/Unlearn implementation\n")
  cat("  scikit-learn + statsmodels               -- from-scratch OLS ANCOVA (see .py)\n")
  cat("Refs: Hansen, B.B. (2008) 'The prognostic analogue of the propensity\n")
  cat("      score', Biometrika 95(2): 481-488; Schuler, A. et al. (2022)\n")
  cat("      'Increasing the efficiency of randomized trial estimates via\n")
  cat("      linear adjustment for a prognostic score', Int J Biostat 18(2);\n")
  cat("      EMA CHMP (2024) qualification opinion for PROCOVA.\n")
}
