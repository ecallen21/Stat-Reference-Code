# Fleming-Harrington weighted log-rank tests (Reference Sec 11.29)
# Native R via survival::survdiff(rho=); Python via custom.
# Run with:  Rscript fleming_harrington_weighted_logrank.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  survival::survdiff(rho=, gamma=)  -- G(rho, gamma) tests\n")
  cat("  FHtest                            -- FH-family and combined-test suite\n")
  cat("  nph                                -- crossing-hazards & max-combo tests\n")
  cat("  survRM2                            -- restricted mean survival alternative\n")
  cat("Python:\n")
  cat("  lifelines.statistics.multivariate_logrank_test (rho baked in)\n")
  cat("  From-scratch (see fleming_harrington_weighted_logrank.py)\n")
  cat("Refs: Fleming, T.R. & Harrington, D.P. (1991) Counting Processes and\n")
  cat("      Survival Analysis, Wiley; Harrington & Fleming (1982) 'A class of\n")
  cat("      rank test procedures for censored survival data', Biometrika 69.\n")
}
