# Cross-classified random-effects model (CCREM) (Reference Sec 8.19)
# Native R via lme4 / glmmTMB; Python via statsmodels MixedLM.
# Run with:  Rscript cross_classified_random_effects.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  lme4::lmer(y ~ 1 + (1 | school) + (1 | neighborhood))\n")
  cat("  glmmTMB::glmmTMB(..., family = ...) -- CCREM for GLMs\n")
  cat("  brms::brm(..., + (1|school) + (1|neigh)) -- Bayesian CCREM\n")
  cat("  MCMCglmm -- Bayesian CCREM with wider families\n")
  cat("Python:\n")
  cat("  statsmodels.regression.mixed_linear_model.MixedLM (VC + formula fallback)\n")
  cat("  pymer4 -- lme4 wrapper via rpy2\n")
  cat("  pymc / numpyro -- Bayesian CCREM\n")
  cat("Refs: Raudenbush, S.W. & Bryk, A.S. (2002) Hierarchical Linear Models,\n")
  cat("      2nd ed., Sage; Goldstein, H. (2011) Multilevel Statistical Models,\n")
  cat("      4th ed., Wiley.\n")
}
