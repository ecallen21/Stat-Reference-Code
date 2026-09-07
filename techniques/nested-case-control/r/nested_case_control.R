# Nested case-control (NCC) study (Reference Sec 15.42)
# Native R via Epi / survival; Python via statsmodels ConditionalLogit.
# Run with:  Rscript nested_case_control.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  Epi::ccwc                  -- draw risk-set-matched controls\n")
  cat("  survival::clogit          -- conditional logistic on matched sets\n")
  cat("  multipleNCC                -- multiple-outcomes / cohort-multiplicity fix\n")
  cat("  mstate                     -- multi-state extensions\n")
  cat("Python:\n")
  cat("  statsmodels.discrete.conditional_models.ConditionalLogit\n")
  cat("  lifelines.CoxPHFitter (for full-cohort comparison)\n")
  cat("Refs: Thomas, D.C. (1977) addendum to Liddell, McDonald & Thomas, JRSS-A;\n")
  cat("      Langholz, B. & Goldstein, L. (1996) 'Risk set sampling in\n")
  cat("      epidemiologic cohort studies', Statistical Science 11(1); Rothman,\n")
  cat("      Greenland & Lash (2020) Modern Epidemiology, 4th ed., ch. 8.\n")
}
