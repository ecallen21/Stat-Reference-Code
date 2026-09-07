# Case-crossover study (Reference Sec 15.36)
# Native R via survival::clogit; Python via statsmodels ConditionalLogit.
# Run with:  Rscript case_crossover.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  survival::clogit             -- conditional logistic (stratum = subject)\n")
  cat("  Epi::clogistic               -- alternative fit\n")
  cat("  survival::coxph(... strata)  -- Cox with strata for time-stratified design\n")
  cat("Python:\n")
  cat("  statsmodels.discrete.conditional_models.ConditionalLogit\n")
  cat("  patsy formulas + stratum indicators\n")
  cat("Refs: Maclure, M. (1991) 'The case-crossover design: a method for studying\n")
  cat("      transient effects on the risk of acute events', Am J Epi 133(2):\n")
  cat("      144-153; Mittleman & Mostofsky (2014) 'Exchangeability in the case-\n")
  cat("      crossover design'.\n")
}
