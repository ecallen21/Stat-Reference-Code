# Exposure crossover / drug-drug interaction (Reference Sec 43.6)
# Native R via survival::clogit, gnm; Python statsmodels + custom.
# Run with:  Rscript exposure_crossover.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  survival::clogit               -- conditional logistic for matched case-crossover\n")
  cat("  gnm                             -- generalised nonlinear models with elim=\n")
  cat("  epiR::epi.interaction           -- RERI + attributable proportion\n")
  cat("Python:\n")
  cat("  statsmodels.discrete.conditional_models.ConditionalLogit\n")
  cat("  custom (RERI + delta-method SE)\n")
  cat("Refs: Hennessy et al. (2014) 'Quality of Medicaid and Medicare data', J Am\n")
  cat("      Health Econ; Rothman (1976) 'The estimation of synergy or antagonism',\n")
  cat("      AJE.\n")
}
