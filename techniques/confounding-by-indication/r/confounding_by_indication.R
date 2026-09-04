# Confounding by indication + protopathic bias (Reference Sec 43.12)
# Native R via MatchIt / WeightIt / CohortMethod; Python causalinference + zepid + dowhy.
# Run with:  Rscript confounding_by_indication.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  MatchIt                          -- new-user matching + covariate balance\n")
  cat("  WeightIt                          -- IPTW with multiple estimation methods\n")
  cat("  cobalt                            -- balance diagnostics\n")
  cat("  CohortMethod (OHDSI)              -- new-user, active-comparator on OMOP CDM\n")
  cat("Python:\n")
  cat("  causalinference                   -- Rubin-style causal-inference toolbox\n")
  cat("  zepid                             -- causal-inference + pharmacoepi\n")
  cat("  dowhy                             -- Microsoft's do-calculus framework\n")
  cat("  custom pandas                     -- new-user cohort construction\n")
  cat("Refs: Salas, Hofman & Stricker (1999) 'Confounding by indication', AJE;\n")
  cat("      Horwitz & Feinstein (1980) 'Protopathic bias', Am J Med.\n")
}
