# Quantile treatment effects (Reference Sec 15.33)
# Native R via Counterfactual / quantreg; Python via econml / from-scratch.
# Run with:  Rscript quantile_treatment_effects.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  Counterfactual::counterfactual  -- Chernozhukov et al. QTE\n")
  cat("  quantreg::rq                     -- conditional QR baseline\n")
  cat("  qte                              -- Callaway QTE for panel / DiD\n")
  cat("Python:\n")
  cat("  econml.dr.LinearDRLearner        -- doubly-robust QTE\n")
  cat("  causalml.metalearners            -- meta-learner QTE\n")
  cat("Refs: Firpo, S. (2007) 'Efficient semiparametric estimation of quantile\n")
  cat("      treatment effects', Econometrica 75(1): 259-276; Chernozhukov &\n")
  cat("      Hansen (2005) 'An IV model of quantile treatment effects'.\n")
}
