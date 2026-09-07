# Hurdle model (Reference Sec 7.20)
# Native R via pscl / hurdlr / countreg; Python via statsmodels / from-scratch.
# Run with:  Rscript hurdle_model.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  pscl::hurdle           -- Poisson / NegBin hurdle with binomial gate\n")
  cat("  countreg::hurdle        -- extended alternative implementations\n")
  cat("  hurdlr                  -- richer Bayesian hurdle families\n")
  cat("  glmmTMB(family = truncated_poisson) -- mixed-effect hurdle\n")
  cat("Python:\n")
  cat("  statsmodels.discrete.count_model.HurdleCountModel\n")
  cat("  pymc / bambi (custom hurdle likelihood)\n")
  cat("  From-scratch scipy (see hurdle_model.py)\n")
  cat("Refs: Mullahy, J. (1986) 'Specification and testing of some modified\n")
  cat("      count data models', J Econ 33; Cragg, J.G. (1971) 'Some\n")
  cat("      statistical models for limited dependent variables with\n")
  cat("      application to the demand for durable goods', Econometrica 39(5).\n")
}
