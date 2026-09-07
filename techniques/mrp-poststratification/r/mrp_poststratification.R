# MRP -- multilevel regression + poststratification (Reference Sec 27.8)
# Native R via brms / rstanarm / lme4; Python via pymc + custom.
# Run with:  Rscript mrp_poststratification.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  rstanarm::stan_glmer + posterior_epred  -- classic MRP workflow\n")
  cat("  brms::brm + posterior_epred              -- flexible MRP with structured priors\n")
  cat("  lme4::glmer (frequentist plug-in)         -- fast approximation\n")
  cat("  MRP (community package)                   -- helpers around poststrat frames\n")
  cat("  survey::svyglm(strata,...) + predict     -- design-based comparison\n")
  cat("Python:\n")
  cat("  pymc + posterior_predict + numpy poststrat  -- Bayesian MRP\n")
  cat("  bambi (pymc wrapper) + poststrat helpers\n")
  cat("  statsmodels.MixedLM (frequentist) + numpy poststrat\n")
  cat("Refs: Gelman, A. & Little, T.C. (1997) 'Poststratification into many\n")
  cat("      categories using hierarchical logistic regression', Survey\n")
  cat("      Methodology 23; Park, D.K., Gelman, A. & Bafumi, J. (2004)\n")
  cat("      'Bayesian multilevel estimation with poststratification', PA 12(4).\n")
}
