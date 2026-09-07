# AIPW / doubly robust (Reference Sec 15.8)
# Native R via AIPW / CausalGAM / tmle; Python DoubleML / EconML.
# Run with:  Rscript aipw_doubly_robust.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  AIPW                              -- AIPW with cross-fitting\n")
  cat("  CausalGAM                          -- GAM-based AIPW\n")
  cat("  tmle                                -- TMLE + AIPW (TMLE targets more efficiency)\n")
  cat("  drtmle                              -- doubly robust TMLE\n")
  cat("Python:\n")
  cat("  DoubleML::DoubleMLIRM              -- interactive regression model (AIPW)\n")
  cat("  econml.dr.DRLearner                -- doubly-robust meta-learner\n")
  cat("  causalinference / zepid + custom\n")
  cat("Refs: Robins, Rotnitzky & Zhao (1994) 'Estimation of regression coefficients\n")
  cat("      when some regressors are not always observed', JASA; Kang & Schafer\n")
  cat("      (2007) 'Demystifying double robustness', Statistical Science.\n")
}
